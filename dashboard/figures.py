import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.theme import ACCENT_COLOR, DOWN_COLOR, UP_COLOR, style_chart


def build_selected_pipeline_comparison_figure(row):
    categories = [
        "Main Native",
        "Aligned Baseline",
        "GIFT",
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Sharpe",
            "F1",
        ],
    )

    fig.add_trace(
        go.Bar(
            x=categories,
            y=[
                row["main_native_sharpe"],
                row["baseline_sharpe"],
                row["gift_sharpe"],
            ],
            name="Sharpe",
            marker_color=[ACCENT_COLOR, "#F59E0B", UP_COLOR],
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=categories,
            y=[
                row["main_native_f1"],
                row["baseline_f1"],
                row["gift_f1"],
            ],
            name="F1",
            marker_color=[ACCENT_COLOR, "#F59E0B", UP_COLOR],
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    style_chart(fig, height=420)

    return fig


def build_comparison_delta_figure(comparison_frame):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=comparison_frame["ticker"],
            y=comparison_frame["gift_minus_baseline_sharpe"],
            name="Sharpe Delta",
            marker_color=[
                UP_COLOR if value >= 0 else DOWN_COLOR
                for value in comparison_frame["gift_minus_baseline_sharpe"]
            ],
        )
    )

    style_chart(fig, height=420)
    fig.update_layout(
        title="GIFT Sharpe Improvement vs Aligned Baseline",
        yaxis_title="Sharpe Delta",
        xaxis_title="Ticker",
    )

    return fig
