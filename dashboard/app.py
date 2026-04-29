import streamlit as st

from dashboard.controller import build_dashboard_state
from dashboard.theme import inject_dashboard_css
from dashboard.views import render_dashboard_tabs


st.set_page_config(
    page_title="Indian Equity ML Signals",
    layout="wide",
)

inject_dashboard_css()

st.title("Indian Equity ML Signal Dashboard")

view_state=build_dashboard_state()
render_dashboard_tabs(view_state)

st.caption("Educational use only")
