from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.figures import (
    build_comparison_delta_figure,
    build_selected_pipeline_comparison_figure,
)
from dashboard.outlook import format_display_date
from dashboard.probability_display import (
    format_prediction_probability,
    format_prediction_probability_markdown,
    format_probability,
    prediction_probability_label,
)
from dashboard.theme import (
    ACCENT_COLOR,
    DOWN_COLOR,
    MUTED_COLOR,
    UP_COLOR,
    prediction_marker_style,
    style_chart,
)
from src.analytics.performance import (
    max_drawdown,
    sharpe_ratio,
    total_return,
    win_rate,
)


@dataclass(frozen=True)
class DashboardViewState:
    analysis_df: pd.DataFrame
    analysis_period_label: str
    analysis_tab: Any
    anchor_date: Any
    benchmark_dd_label: str
    benchmark_label: str
    bt: pd.DataFrame
    cls_metrics: dict[str, float]
    cls_metrics_note: str
    comparison_frame: pd.DataFrame
    comparison_tab: Any
    entry_plot_column: str
    entry_points: pd.DataFrame
    equity: pd.Series
    exit_plot_column: str
    exit_points: pd.DataFrame
    index_dd_label: str
    index_label: str
    latest_probability_label: str
    latest_probability_value: float
    latest_row: Any
    merged: pd.DataFrame
    nifty_dd: pd.Series
    period_end: Any
    period_start: Any
    pipeline_key: str
    returns: pd.Series
    signal_policy: dict[str, float]
    stock_dd: pd.Series
    strategy_dd: pd.Series
    strategy_period_caption: str
    three_day_outlook: pd.DataFrame
    ticker: str
    trade_hit_rate: float
    trade_mask: pd.Series


def render_dashboard_tabs(state: DashboardViewState) -> None:
    analysis_df=state.analysis_df
    analysis_period_label=state.analysis_period_label
    analysis_tab=state.analysis_tab
    anchor_date=state.anchor_date
    benchmark_dd_label=state.benchmark_dd_label
    benchmark_label=state.benchmark_label
    bt=state.bt
    cls_metrics=state.cls_metrics
    cls_metrics_note=state.cls_metrics_note
    comparison_frame=state.comparison_frame
    comparison_tab=state.comparison_tab
    entry_plot_column=state.entry_plot_column
    entry_points=state.entry_points
    equity=state.equity
    exit_plot_column=state.exit_plot_column
    exit_points=state.exit_points
    index_dd_label=state.index_dd_label
    index_label=state.index_label
    latest_probability_label=state.latest_probability_label
    latest_probability_value=state.latest_probability_value
    latest_row=state.latest_row
    merged=state.merged
    nifty_dd=state.nifty_dd
    period_end=state.period_end
    period_start=state.period_start
    pipeline_key=state.pipeline_key
    returns=state.returns
    signal_policy=state.signal_policy
    stock_dd=state.stock_dd
    strategy_dd=state.strategy_dd
    strategy_period_caption=state.strategy_period_caption
    three_day_outlook=state.three_day_outlook
    ticker=state.ticker
    trade_hit_rate=state.trade_hit_rate
    trade_mask=state.trade_mask

    with analysis_tab:

        st.subheader("Latest Snapshot")

        s1,s2,s3,s4=st.columns(4)

        s1.metric("Latest Close",f"₹{latest_row.Close:,.2f}")
        s2.metric(latest_probability_label,format_probability(latest_probability_value))
        s3.metric("Classifier Call",latest_row.classifier_prediction)
        s4.metric("Trade State",latest_row.trade_state)

        st.caption(
            f"Selected period {analysis_period_label}. "
            f"Classifier threshold {cls_metrics['threshold']:.2f}. "
            f"Trade entry / exit {signal_policy['entry_threshold']:.2f} / "
            f"{signal_policy['exit_threshold']:.2f}. "
            "DOWN calls display P(Down), calculated as 1 - P(Up)."
        )

        st.subheader("Price Curve")

        price_fig=go.Figure()

        price_fig.add_trace(go.Scatter(
            x=analysis_df["Date"],
            y=analysis_df["Close"],
            mode="lines",
            name="Close",
            line=dict(color=ACCENT_COLOR,width=2.4),
            hovertemplate="Close: ₹%{y:,.2f}<extra></extra>",
        ))

        for prediction in ["UP","DOWN"]:
            prediction_points=analysis_df.loc[
                analysis_df["classifier_prediction"]==prediction
            ].copy()

            if prediction_points.empty:
                continue

            probability_label=prediction_probability_label(prediction)

            price_fig.add_trace(go.Scatter(
                x=prediction_points["Date"],
                y=prediction_points["Close"],
                mode="markers",
                marker=prediction_marker_style(prediction),
                name=f"{prediction} prediction",
                customdata=prediction_points[["classifier_probability"]],
                hovertemplate=(
                    f"{prediction} prediction<br>"
                    "Close: ₹%{y:,.2f}<br>"
                    f"{probability_label}: %{{customdata[0]:.2%}}"
                    "<extra></extra>"
                ),
            ))

        if not entry_points.empty:

            price_fig.add_trace(go.Scatter(
                x=entry_points["Date"],
                y=entry_points[entry_plot_column],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=9,
                    color="white",
                    line=dict(color=UP_COLOR,width=2),
                ),
                name="Trade entry",
                hovertemplate="Trade entry<br>Price: ₹%{y:,.2f}<extra></extra>",
            ))

        if not exit_points.empty:

            price_fig.add_trace(go.Scatter(
                x=exit_points["Date"],
                y=exit_points[exit_plot_column],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=9,
                    color="white",
                    line=dict(color=DOWN_COLOR,width=2),
                ),
                name="Trade exit",
                hovertemplate="Trade exit<br>Price: ₹%{y:,.2f}<extra></extra>",
            ))

        style_chart(price_fig, height=500)

        st.plotly_chart(price_fig,width="stretch")


        st.subheader("Probability Curve")

        prob_fig=go.Figure()

        prob_fig.add_trace(go.Scatter(
            x=analysis_df["Date"],
            y=analysis_df["probability_up"],
            mode="lines",
            name="P(Up)",
            line=dict(color=ACCENT_COLOR,width=2.2),
            hovertemplate="P(Up): %{y:.2%}<extra></extra>",
        ))

        prob_fig.add_hline(
            y=float(signal_policy["entry_threshold"]),
            line_dash="dash",
            line_color=UP_COLOR,
            annotation_text="Trade Entry",
        )

        if float(signal_policy["exit_threshold"])!=float(signal_policy["entry_threshold"]):
            prob_fig.add_hline(
                y=float(signal_policy["exit_threshold"]),
                line_dash="dash",
                line_color=DOWN_COLOR,
                annotation_text="Trade Exit",
            )

        if not np.isclose(
            float(cls_metrics["threshold"]),
            float(signal_policy["entry_threshold"]),
        ):
            prob_fig.add_hline(
                y=float(cls_metrics["threshold"]),
                line_dash="dot",
                line_color=MUTED_COLOR,
                annotation_text="Classifier Threshold",
            )

        style_chart(prob_fig, height=420, yaxis_tickformat=".0%")
        prob_fig.update_yaxes(range=[0,1])

        st.plotly_chart(prob_fig,width="stretch")


        if not three_day_outlook.empty:

            st.subheader("3-Day Price Outlook")

            if pipeline_key=="gift":

                st.caption(
                    "For forward projected sessions, the GIFT-aware outlook uses the "
                    "latest available GIFT snapshot at or before the forecast date."
                )

            st.write(
                "Period: "
                f"{format_display_date(period_start,'%Y-%m-%d')} to "
                f"{format_display_date(period_end,'%Y-%m-%d')} | "
                "Anchor: "
                f"{format_display_date(anchor_date,'%Y-%m-%d')}"
            )

            outlook_columns=st.columns(len(three_day_outlook))

            for column,row in zip(outlook_columns,three_day_outlook.itertuples()):

                column.metric(
                    row.session,
                    f"₹{row.price:,.2f}",
                    "N/A" if pd.isna(row.change_pct) else f"{row.change_pct:+.2%}",
                )

                column.write(format_display_date(row.date))

                if pd.isna(row.probability_up):
                    column.write("Prediction: `N/A`")
                    column.write("P(Up): `N/A`")
                else:
                    column.write(f"Prediction: `{row.prediction}`")
                    column.write(
                        format_prediction_probability_markdown(
                            row.probability_up,
                            row.prediction,
                        )
                    )

                column.write(f"Check: `{row.verification}`")

            display_outlook=three_day_outlook.copy()

            display_outlook["Date"]=display_outlook["date"].map(
                lambda value:format_display_date(value,"%Y-%m-%d")
            )
            display_outlook["Price"]=display_outlook["price"].map(
                lambda value:f"₹{value:,.2f}"
            )
            display_outlook["Move vs Prior Close"]=display_outlook["change_pct"].map(
                lambda value:"N/A" if pd.isna(value) else f"{value:+.2%}"
            )
            display_outlook["Model Probability"]=display_outlook.apply(
                lambda row:format_prediction_probability(
                    row["probability_up"],
                    row["prediction"],
                ),
                axis=1,
            )
            display_outlook["Index Close"]=display_outlook["index_close"].map(
                lambda value:"N/A" if pd.isna(value) else f"{value:,.2f}"
            )
            display_outlook["Index Move"]=display_outlook["index_change_pct"].map(
                lambda value:"N/A" if pd.isna(value) else f"{value:+.2%}"
            )

            st.dataframe(
                display_outlook.rename(columns={
                    "session":"Session",
                    "prediction":"Model Call",
                    "basis":"Price Basis",
                    "realized_move":"Actual Move",
                    "verification":"Verification",
                })[[
                    "Session",
                    "Date",
                    "Price",
                    "Move vs Prior Close",
                    "Model Call",
                    "Model Probability",
                    "Actual Move",
                    "Verification",
                    "Index Close",
                    "Index Move",
                    "Price Basis",
                ]],
                hide_index=True,
                width="stretch",
            )


        st.subheader("Strategy Performance")

        st.caption(strategy_period_caption)

        c1,c2,c3,c4=st.columns(4)

        total_return_value=total_return(equity) if not equity.empty else 0.0
        sharpe_value=sharpe_ratio(returns) if not returns.empty else 0.0
        max_dd_value=max_drawdown(equity) if not equity.empty else 0.0
        win_rate_value=win_rate(returns) if not returns.empty else 0.0

        c1.metric("Total Return",f"{total_return_value:.2%}")
        c2.metric("Sharpe",f"{sharpe_value:.2f}")
        c3.metric("Max DD",f"{max_dd_value:.2%}")
        c4.metric("Win Rate",f"{win_rate_value:.2%}")

        st.caption(
            f"Long-signal hit rate {trade_hit_rate:.2%} across "
            f"{int(trade_mask.sum())} active sessions."
        )

        st.subheader("Equity Curve")

        fig=go.Figure()

        fig.add_trace(go.Scatter(
            x=bt["Date"],
            y=equity,
            mode="lines",
            name="Strategy",
            line=dict(color=ACCENT_COLOR,width=2.4),
        ))

        style_chart(fig, height=420)

        st.plotly_chart(fig,width="stretch")


        st.subheader("Strategy vs Benchmarks")

        if merged.empty:

            st.warning("No overlapping benchmark rows are available for the selected period")

        else:

            comp=go.Figure()

            comp.add_trace(go.Scatter(
                x=merged["Date"],
                y=merged["strategy_norm"],
                mode="lines",
                name="Strategy",
                line=dict(color=ACCENT_COLOR,width=2.4),
            ))

            comp.add_trace(go.Scatter(
                x=merged["Date"],
                y=merged["stock_norm"],
                mode="lines",
                name=benchmark_label,
                line=dict(color="#F59E0B",width=2),
            ))

            comp.add_trace(go.Scatter(
                x=merged["Date"],
                y=merged["nifty_norm"],
                mode="lines",
                name=index_label,
                line=dict(color=UP_COLOR,width=2),
            ))

            style_chart(comp, height=430)

            st.plotly_chart(comp,width="stretch")


        st.subheader("Drawdown Comparison")

        if merged.empty:

            st.warning("Drawdown comparison is unavailable without overlapping NIFTY data")

        else:

            dd_fig=go.Figure()

            dd_fig.add_trace(go.Scatter(
                x=merged["Date"],
                y=strategy_dd,
                mode="lines",
                name="Strategy DD",
                line=dict(color=ACCENT_COLOR,width=2.2),
            ))

            dd_fig.add_trace(go.Scatter(
                x=merged["Date"],
                y=stock_dd,
                mode="lines",
                name=benchmark_dd_label,
                line=dict(color="#F59E0B",width=2),
            ))

            dd_fig.add_trace(go.Scatter(
                x=merged["Date"],
                y=nifty_dd,
                mode="lines",
                name=index_dd_label,
                line=dict(color=UP_COLOR,width=2),
            ))

            style_chart(dd_fig, height=430, yaxis_tickformat=".0%")

            st.plotly_chart(dd_fig,width="stretch")


        st.subheader("Classifier Summary")

        m1,m2,m3=st.columns(3)

        m1.metric("F1",f"{cls_metrics['f1']:.2%}")
        m2.metric("Precision",f"{cls_metrics['precision']:.2%}")
        m3.metric("Recall",f"{cls_metrics['recall']:.2%}")

        st.caption(
            f"{cls_metrics_note} Threshold {cls_metrics['threshold']:.2f}. "
            f"Positive prediction rate {cls_metrics['positive_rate']:.2%} "
            f"across {cls_metrics['support']} rows."
        )

        st.subheader("Recent Model Data")

        recent_view=analysis_df.tail(60).copy()
        recent_view["Date"]=recent_view["Date"].map(
            lambda value:format_display_date(value,"%Y-%m-%d")
        )
        recent_view["Close"]=recent_view["Close"].map(
            lambda value:f"₹{value:,.2f}"
        )
        recent_view["model_probability"]=recent_view.apply(
            lambda row:format_prediction_probability(
                row["probability_up"],
                row["classifier_prediction"],
            ),
            axis=1,
        )

        st.dataframe(
            recent_view.rename(columns={
                "model_probability":"Model Probability",
                "classifier_prediction":"Classifier Call",
                "trade_state":"Trade State",
                "realized_direction":"Actual",
            })[[
                "Date",
                "Close",
                "Classifier Call",
                "Model Probability",
                "Trade State",
                "Actual",
            ]],
            hide_index=True,
            width="stretch",
        )

    with comparison_tab:

        if comparison_frame.empty:

            st.info("No pipeline comparison data is available.")

        else:

            ticker_comparison=comparison_frame.loc[
                comparison_frame["ticker"]==ticker
            ].copy()

            if ticker_comparison.empty:

                st.info("No pipeline comparison data is available for the selected ticker.")

            else:

                comparison_row=ticker_comparison.iloc[0]

                st.subheader("Pipeline Comparison")

                st.caption(
                    "Main native metrics use the original dashboard pipeline artifacts. "
                    "Aligned baseline retrains the intraday task without GIFT features. "
                    "GIFT metrics come from the standalone GIFT-aware pipeline over "
                    f"{comparison_row['comparison_start_date']} to "
                    f"{comparison_row['comparison_end_date']}."
                )

                p1,p2,p3,p4=st.columns(4)

                p1.metric(
                    "Main Sharpe",
                    f"{comparison_row['main_native_sharpe']:.2f}",
                )

                p2.metric(
                    "Aligned Baseline Sharpe",
                    f"{comparison_row['baseline_sharpe']:.2f}",
                )

                p3.metric(
                    "GIFT Sharpe",
                    f"{comparison_row['gift_sharpe']:.2f}",
                )

                p4.metric(
                    "GIFT Sharpe Delta",
                    f"{comparison_row['gift_minus_baseline_sharpe']:+.2f}",
                )

                q1,q2,q3=st.columns(3)

                q1.metric(
                    "Main F1",
                    f"{comparison_row['main_native_f1']:.2%}",
                )

                q2.metric(
                    "GIFT F1",
                    f"{comparison_row['gift_f1']:.2%}",
                )

                q3.metric(
                    "GIFT F1 Delta",
                    f"{comparison_row['gift_minus_baseline_f1']:+.2%}",
                )

                st.plotly_chart(
                    build_selected_pipeline_comparison_figure(
                        comparison_row
                    ),
                    width="stretch",
                )

                overall_sharpe_wins=int(
                    (comparison_frame["gift_minus_baseline_sharpe"]>0).sum()
                )
                overall_f1_wins=int(
                    (comparison_frame["gift_minus_baseline_f1"]>0).sum()
                )

                r1,r2,r3,r4=st.columns(4)

                r1.metric(
                    "Sharpe Wins",
                    f"{overall_sharpe_wins}/{len(comparison_frame)}",
                )

                r2.metric(
                    "F1 Wins",
                    f"{overall_f1_wins}/{len(comparison_frame)}",
                )

                r3.metric(
                    "Avg Sharpe Delta",
                    f"{comparison_frame['gift_minus_baseline_sharpe'].mean():+.2f}",
                )

                r4.metric(
                    "Avg F1 Delta",
                    f"{comparison_frame['gift_minus_baseline_f1'].mean():+.2%}",
                )

                st.plotly_chart(
                    build_comparison_delta_figure(comparison_frame),
                    width="stretch",
                )

                st.dataframe(
                    ticker_comparison[[
                        "ticker",
                        "main_native_model",
                        "main_native_sharpe",
                        "baseline_model",
                        "baseline_sharpe",
                        "gift_model",
                        "gift_sharpe",
                        "gift_minus_baseline_sharpe",
                        "main_native_f1",
                        "gift_f1",
                        "gift_minus_baseline_f1",
                    ]].rename(columns={
                        "ticker":"Ticker",
                        "main_native_model":"Main Model",
                        "main_native_sharpe":"Main Sharpe",
                        "baseline_model":"Aligned Baseline Model",
                        "baseline_sharpe":"Aligned Baseline Sharpe",
                        "gift_model":"GIFT Model",
                        "gift_sharpe":"GIFT Sharpe",
                        "gift_minus_baseline_sharpe":"GIFT Sharpe Delta",
                        "main_native_f1":"Main F1",
                        "gift_f1":"GIFT F1",
                        "gift_minus_baseline_f1":"GIFT F1 Delta",
                    }),
                    hide_index=True,
                    width="stretch",
                )
