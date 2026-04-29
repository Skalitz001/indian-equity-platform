import streamlit as st


UP_COLOR = "#0F9D58"
DOWN_COLOR = "#D72638"
ACCENT_COLOR = "#1B4D89"
MUTED_COLOR = "#6B7280"
PLOT_BACKGROUND = "rgba(255, 255, 255, 0)"
GRID_COLOR = "rgba(35, 48, 70, 0.10)"
CHART_COLORWAY = [
    ACCENT_COLOR,
    UP_COLOR,
    DOWN_COLOR,
    "#F59E0B",
    "#64748B",
]


def prediction_marker_style(prediction, size=8):
    is_down = prediction == "DOWN"

    return {
        "symbol": "triangle-down" if is_down else "triangle-up",
        "size": size,
        "color": DOWN_COLOR if is_down else UP_COLOR,
        "opacity": 0.78,
        "line": {"width": 1, "color": "white"},
    }


def style_chart(fig, *, height=430, yaxis_tickformat=None):
    fig.update_layout(
        template="plotly_white",
        colorway=CHART_COLORWAY,
        height=height,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=28, r=24, t=56, b=34),
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font=dict(color="#1F2937"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)

    if yaxis_tickformat:
        fig.update_yaxes(tickformat=yaxis_tickformat)

    return fig


def inject_dashboard_css():
    st.markdown(
        """
        <style>
        :root {
            --surface: rgba(255, 255, 255, 0.88);
            --border: rgba(31, 41, 55, 0.10);
            --ink: #172033;
            --muted: #64748b;
            --up: #0f9d58;
            --down: #d72638;
            --accent: #1b4d89;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(27, 77, 137, 0.13), transparent 30rem),
                radial-gradient(circle at 88% 12%, rgba(15, 157, 88, 0.12), transparent 26rem),
                linear-gradient(135deg, #f7faf7 0%, #eef4f8 52%, #f8f4ed 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 2.5rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.035em;
        }

        h1 {
            font-size: clamp(2rem, 3.5vw, 3.4rem) !important;
            line-height: 0.95 !important;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(241, 246, 247, 0.92)),
                radial-gradient(circle at top left, rgba(27, 77, 137, 0.16), transparent 18rem);
            border-right: 1px solid var(--border);
        }

        [data-testid="stMetric"],
        [data-testid="stDataFrame"],
        .stPlotlyChart {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: 0 18px 55px rgba(31, 41, 55, 0.08);
            padding: 0.85rem;
        }

        [data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-weight: 650;
        }

        [data-testid="stTabs"] button {
            border-radius: 999px;
            padding: 0.35rem 1rem;
        }

        [data-testid="stCaptionContainer"] {
            color: var(--muted);
        }

        div[data-baseweb="select"] > div,
        [data-testid="stDateInput"] input {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
