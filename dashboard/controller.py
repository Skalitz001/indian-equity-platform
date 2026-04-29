import json

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.outlook import (
    build_three_day_outlook,
    build_three_day_outlook_gift,
    format_display_date,
)
from dashboard.pipeline import (
    add_intraday_returns,
    artifacts_dir_for_pipeline,
    build_feature_frame,
    dashboard_tickers,
    load_comparison_summary,
    load_prediction_model,
    meta_path_for_pipeline,
    resolve_classifier_metrics,
)
from dashboard.probability_display import (
    prediction_label,
    prediction_probability,
    prediction_probability_label,
)
from dashboard.views import DashboardViewState
from src.analytics.experimental_features import resolve_main_feature_group
from src.analytics.features import add_log_returns
from src.backtesting.engine import BacktestEngine
from src.gift_nifty.backtesting import IntradayBacktestEngine
from src.gift_nifty.constants import DEFAULT_GIFT_DATA_PATH
from src.gift_nifty.dataset import build_gift_model_frame
from src.gift_nifty.repository import GiftNiftyRepository
from src.models.probabilities import predict_up_probability
from src.repositories.market_data_repository import MarketDataRepository
from src.strategies.signal_policy import (
    generate_probability_signals,
    normalize_signal_policy,
)


def _selected_pipeline() -> str:
    pipeline_choice=st.sidebar.selectbox(
        "Pipeline",
        [
            "Main Pipeline",
            "GIFT-Aware Pipeline",
        ],
    )

    return (
        "gift"
        if pipeline_choice=="GIFT-Aware Pipeline"
        else "main"
    )


def _load_model_metadata(ticker: str, pipeline_key: str) -> tuple[dict, list[str]]:
    meta_path=meta_path_for_pipeline(
        ticker,
        pipeline_key,
    )

    if not meta_path.exists():
        st.error("Model not trained for the selected pipeline")
        st.stop()

    with open(meta_path) as f:
        meta=json.load(f)

    return meta,meta["features"]


def _selected_model(meta: dict, artifacts_dir):
    model_options=[
        "Logistic Regression",
        "Random Forest",
    ]

    if "blend_weights" in meta:
        model_options.append("Ensemble")

    recommended_option={
        "logistic":"Logistic Regression",
        "rf":"Random Forest",
        "ensemble":"Ensemble",
    }.get(meta.get("recommended_model"),"Logistic Regression")

    default_model_index=(
        model_options.index(recommended_option)
        if recommended_option in model_options else 0
    )

    model_choice=st.sidebar.selectbox(
        "Model",
        model_options,
        index=default_model_index,
    )

    selected_model_name={
        "Logistic Regression":"logistic",
        "Random Forest":"rf",
        "Ensemble":"ensemble",
    }[model_choice]

    if selected_model_name=="ensemble" and "blend_weights" not in meta:
        selected_model_name="logistic"

    try:
        model=load_prediction_model(
            meta,
            selected_model_name,
            artifacts_dir,
        )
    except FileNotFoundError:
        st.error("Model file missing")
        st.stop()

    return model,selected_model_name


def _load_gift_history() -> pd.DataFrame:
    try:
        gift_history=GiftNiftyRepository(DEFAULT_GIFT_DATA_PATH).load()
    except FileNotFoundError:
        st.error(
            "GIFT history is missing. Run "
            "`python3 -m src.gift_nifty.ingest_dhan` first."
        )
        st.stop()

    return gift_history.sort_values("Date").reset_index(drop=True)


def _build_pipeline_frame(
    pipeline_key: str,
    price_history: pd.DataFrame,
    gift_history: pd.DataFrame,
    main_feature_group: str,
) -> pd.DataFrame:
    if pipeline_key=="gift":
        df=build_gift_model_frame(
            price_history,
            gift_history,
        )

        if df.empty:
            st.error("Not enough GIFT-aligned data to build the selected asset")
            st.stop()

        return df

    df=build_feature_frame(
        price_history,
        feature_group=main_feature_group,
    ).copy()

    next_log_return=df["log_return"].shift(-1)

    df["target"]=np.where(
        next_log_return.notna(),
        (next_log_return>0).astype(int),
        np.nan,
    )

    df=df.dropna().reset_index(drop=True)

    if df.empty:
        st.error("Not enough data to build model features for this asset")
        st.stop()

    return df


def _pipeline_labels(pipeline_key: str):
    if pipeline_key=="gift":
        return {
            "description":(
                "GIFT-aware pipeline: same-day open-to-close classification "
                "using previous stock state plus pre-open GIFT features."
            ),
            "benchmark_label":"Buy & Hold Intraday",
            "benchmark_dd_label":"Buy & Hold Intraday DD",
            "index_label":"NIFTY50 Intraday",
            "index_dd_label":"NIFTY50 Intraday DD",
        }

    return {
        "description":(
            "Main pipeline: next-day close-to-close classification with "
            "next-bar execution."
        ),
        "benchmark_label":"Buy & Hold",
        "benchmark_dd_label":"Buy & Hold DD",
        "index_label":"NIFTY50",
        "index_dd_label":"NIFTY50 DD",
    }


def _selected_analysis_window(
    repo: MarketDataRepository,
    df: pd.DataFrame,
    price_history: pd.DataFrame,
):
    analysis_min_date=pd.Timestamp(df["Date"].min()).date()
    analysis_max_date=pd.Timestamp(df["Date"].max()).date()

    default_period_start=max(
        analysis_min_date,
        (pd.Timestamp(analysis_max_date)-pd.DateOffset(years=3)).date(),
    )

    period_start=st.sidebar.date_input(
        "Period Start",
        value=default_period_start,
        min_value=analysis_min_date,
        max_value=analysis_max_date,
    )

    period_end=st.sidebar.date_input(
        "Period End",
        value=analysis_max_date,
        min_value=analysis_min_date,
        max_value=analysis_max_date,
    )

    if period_start>period_end:
        st.error("Period start must be earlier than or equal to period end")
        st.stop()

    analysis_df=repo.slice_between(
        df,
        start_date=period_start,
        end_date=period_end,
    )
    analysis_df=analysis_df.copy()

    if analysis_df.empty:
        st.error("No model-ready rows are available inside the selected period")
        st.stop()

    anchor_candidates=repo.slice_between(
        price_history,
        start_date=period_start,
        end_date=period_end,
    )

    if anchor_candidates.empty:
        st.error("No trading sessions are available inside the selected period")
        st.stop()

    requested_anchor_date=st.sidebar.date_input(
        "Outlook Anchor Date",
        value=pd.Timestamp(anchor_candidates["Date"].max()).date(),
        min_value=period_start,
        max_value=period_end,
    )

    anchor_date=repo.nearest_available_date(
        anchor_candidates,
        requested_anchor_date,
        direction="backward",
    )

    if pd.isna(anchor_date):
        anchor_date=repo.nearest_available_date(
            anchor_candidates,
            requested_anchor_date,
            direction="forward",
        )

    if pd.isna(anchor_date):
        st.error("Unable to resolve a trading session for the selected outlook date")
        st.stop()

    if pd.Timestamp(requested_anchor_date).date()!=pd.Timestamp(anchor_date).date():
        st.info(
            "Using "
            f"{format_display_date(anchor_date,'%Y-%m-%d')} "
            "because the selected calendar date is not a trading session."
        )

    return period_start,period_end,analysis_df,anchor_date


def _apply_signals(
    df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    meta: dict,
    model_key: str,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    signal_policy=normalize_signal_policy(
        meta.get("signal_policies",{}).get(model_key),
        fallback_threshold=float(
            meta.get("thresholds",{}).get(
                model_key,
                meta.get("opt_threshold",0.55),
            )
        ),
    )

    df["signal"]=generate_probability_signals(
        df["probability_up"],
        signal_policy,
    ).astype(int).to_numpy()
    df["signal_change"]=df["signal"].diff().fillna(df["signal"])

    signal_lookup=df.set_index("model_row_id")["signal"]
    signal_change_lookup=df.set_index("model_row_id")["signal_change"]

    analysis_df["signal"]=(
        analysis_df["model_row_id"].map(signal_lookup).fillna(0).astype(int)
    )
    analysis_df["signal_change"]=(
        analysis_df["model_row_id"].map(signal_change_lookup).fillna(0.0)
    )

    signals=pd.Series(analysis_df["signal"].to_numpy(),index=analysis_df.index)

    return analysis_df,signals,signal_policy


def _add_classifier_columns(
    analysis_df: pd.DataFrame,
    cls_metrics: dict,
) -> pd.DataFrame:
    analysis_df["trade_state"]=np.where(
        analysis_df["signal"]==1,
        "LONG",
        "FLAT",
    )
    analysis_df["classifier_prediction"]=analysis_df["probability_up"].map(
        lambda value:prediction_label(
            float(value),
            cls_metrics["threshold"],
        )
    )
    analysis_df["classifier_probability"]=analysis_df.apply(
        lambda row:prediction_probability(
            row["probability_up"],
            row["classifier_prediction"],
        ),
        axis=1,
    )
    analysis_df["classifier_probability_label"]=analysis_df[
        "classifier_prediction"
    ].map(prediction_probability_label)
    analysis_df["realized_direction"]=analysis_df["target"].map(
        lambda value:"UP" if int(value)==1 else "DOWN"
    )

    return analysis_df


def _entry_exit_points(analysis_df: pd.DataFrame, pipeline_key: str):
    if pipeline_key=="gift":
        analysis_df["stock_equity"]=(1+analysis_df["intraday_return"]).cumprod()
        entry_points=analysis_df.loc[analysis_df["signal"]==1].copy()
        exit_points=analysis_df.loc[analysis_df["signal"]==1].copy()
        return entry_points,exit_points,"Open","Close"

    analysis_df["stock_equity"]=(1+analysis_df["log_return"]).cumprod()
    entry_points=analysis_df.loc[analysis_df["signal_change"]>0].copy()
    exit_points=analysis_df.loc[analysis_df["signal_change"]<0].copy()
    return entry_points,exit_points,"Close","Close"


def _benchmark_merge(
    repo: MarketDataRepository,
    pipeline_key: str,
    engine,
    df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    nifty_history: pd.DataFrame,
    period_start,
    period_end,
):
    full_signals=pd.Series(df["signal"].to_numpy(),index=df.index)
    full_bt=engine.run(df,full_signals)
    bt=repo.slice_between(
        full_bt,
        start_date=period_start,
        end_date=period_end,
    ).copy()
    bt["equity_curve"]=(1+bt["strategy_return"].fillna(0.0)).cumprod()

    nifty=repo.slice_between(
        nifty_history,
        start_date=analysis_df["Date"].min(),
        end_date=analysis_df["Date"].max(),
    )

    if pipeline_key=="gift":
        nifty=add_intraday_returns(nifty)
        nifty=nifty[["Date","intraday_return"]]
        nifty=nifty.dropna()
        nifty["nifty_equity"]=(1+nifty["intraday_return"]).cumprod()
    else:
        nifty=add_log_returns(nifty)
        nifty=nifty[["Date","log_return"]]
        nifty=nifty.dropna()
        nifty["nifty_equity"]=(1+nifty["log_return"]).cumprod()

    merged=pd.merge(
        bt[["Date","equity_curve"]],
        analysis_df[["Date","stock_equity"]],
        on="Date",
        how="inner",
    )

    merged=pd.merge(
        merged,
        nifty,
        on="Date",
        how="inner",
    )

    if not merged.empty:
        merged["strategy_norm"]=merged["equity_curve"]/merged["equity_curve"].iloc[0]
        merged["stock_norm"]=merged["stock_equity"]/merged["stock_equity"].iloc[0]
        merged["nifty_norm"]=merged["nifty_equity"]/merged["nifty_equity"].iloc[0]

        strategy_dd=(
            merged["strategy_norm"]
            /
            merged["strategy_norm"].cummax()
        )-1

        stock_dd=(
            merged["stock_norm"]
            /
            merged["stock_norm"].cummax()
        )-1

        nifty_dd=(
            merged["nifty_norm"]
            /
            merged["nifty_norm"].cummax()
        )-1
    else:
        strategy_dd=pd.Series(dtype=float)
        stock_dd=pd.Series(dtype=float)
        nifty_dd=pd.Series(dtype=float)

    return bt,merged,strategy_dd,stock_dd,nifty_dd


def build_dashboard_state() -> DashboardViewState:
    repo=MarketDataRepository()
    main_engine=BacktestEngine(transaction_cost=0.001)
    gift_engine=IntradayBacktestEngine(transaction_cost=0.001)

    st.sidebar.header("Configuration")

    pipeline_key=_selected_pipeline()

    tickers=dashboard_tickers(repo,pipeline_key)

    if not tickers:
        st.error("No trained assets are available for the selected pipeline")
        st.stop()

    ticker=st.sidebar.selectbox(
        "Select Asset",
        tickers,
    )

    meta,features=_load_model_metadata(ticker,pipeline_key)
    main_feature_group=(
        resolve_main_feature_group(
            meta=meta,
            features=features,
        )
        if pipeline_key=="main"
        else "baseline"
    )
    artifacts_dir=artifacts_dir_for_pipeline(pipeline_key)
    model,model_key=_selected_model(meta,artifacts_dir)

    price_history=repo.load(ticker)
    price_history=price_history.sort_values("Date").reset_index(drop=True)
    nifty_history=repo.load("^NSEI")
    nifty_history=nifty_history.sort_values("Date").reset_index(drop=True)

    gift_history=(
        _load_gift_history()
        if pipeline_key=="gift"
        else pd.DataFrame()
    )

    labels=_pipeline_labels(pipeline_key)
    engine=gift_engine if pipeline_key=="gift" else main_engine

    df=_build_pipeline_frame(
        pipeline_key,
        price_history,
        gift_history,
        main_feature_group,
    )

    st.caption(labels["description"])

    x=df[features]
    df["probability_up"]=predict_up_probability(model,x)
    df["model_row_id"]=np.arange(len(df))

    period_start,period_end,analysis_df,anchor_date=_selected_analysis_window(
        repo,
        df,
        price_history,
    )

    analysis_df,signals,signal_policy=_apply_signals(
        df,
        analysis_df,
        meta,
        model_key,
    )

    three_day_outlook=(
        build_three_day_outlook(
            price_history,
            nifty_history,
            df,
            model,
            features,
            anchor_date,
            feature_group=main_feature_group,
        )
        if pipeline_key=="main"
        else build_three_day_outlook_gift(
            price_history,
            gift_history,
            nifty_history,
            df,
            model,
            features,
            anchor_date,
        )
    )

    cls_metrics,cls_metrics_note=resolve_classifier_metrics(
        meta,
        model_key,
        analysis_df,
    )

    analysis_df=_add_classifier_columns(
        analysis_df,
        cls_metrics,
    )

    (
        entry_points,
        exit_points,
        entry_plot_column,
        exit_plot_column,
    )=_entry_exit_points(analysis_df,pipeline_key)

    latest_row=analysis_df.iloc[-1]
    latest_probability_label=prediction_probability_label(
        latest_row.classifier_prediction
    )
    latest_probability_value=prediction_probability(
        latest_row.probability_up,
        latest_row.classifier_prediction,
    )

    trade_mask=signals==1
    trade_hit_rate=(
        (analysis_df.loc[trade_mask,"target"]==1).mean()
        if trade_mask.sum()>0
        else 0
    )

    bt,merged,strategy_dd,stock_dd,nifty_dd=_benchmark_merge(
        repo,
        pipeline_key,
        engine,
        df,
        analysis_df,
        nifty_history,
        period_start,
        period_end,
    )

    returns=bt["strategy_return"].dropna()
    equity=bt["equity_curve"]

    analysis_period_label=(
        f"{format_display_date(analysis_df['Date'].min(),'%Y-%m-%d')} to "
        f"{format_display_date(analysis_df['Date'].max(),'%Y-%m-%d')}"
    )
    strategy_period_caption=(
        "Trading strategy returns period: "
        f"{analysis_period_label} across {len(analysis_df)} trading sessions."
    )
    comparison_frame=load_comparison_summary()

    analysis_tab,comparison_tab=st.tabs([
        "Signal & Performance",
        "Comparison",
    ])

    return DashboardViewState(
        analysis_df=analysis_df,
        analysis_period_label=analysis_period_label,
        analysis_tab=analysis_tab,
        anchor_date=anchor_date,
        benchmark_dd_label=labels["benchmark_dd_label"],
        benchmark_label=labels["benchmark_label"],
        bt=bt,
        cls_metrics=cls_metrics,
        cls_metrics_note=cls_metrics_note,
        comparison_frame=comparison_frame,
        comparison_tab=comparison_tab,
        entry_plot_column=entry_plot_column,
        entry_points=entry_points,
        equity=equity,
        exit_plot_column=exit_plot_column,
        exit_points=exit_points,
        index_dd_label=labels["index_dd_label"],
        index_label=labels["index_label"],
        latest_probability_label=latest_probability_label,
        latest_probability_value=latest_probability_value,
        latest_row=latest_row,
        merged=merged,
        nifty_dd=nifty_dd,
        period_end=period_end,
        period_start=period_start,
        pipeline_key=pipeline_key,
        returns=returns,
        signal_policy=signal_policy,
        stock_dd=stock_dd,
        strategy_dd=strategy_dd,
        strategy_period_caption=strategy_period_caption,
        three_day_outlook=three_day_outlook,
        ticker=ticker,
        trade_hit_rate=trade_hit_rate,
        trade_mask=trade_mask,
    )
