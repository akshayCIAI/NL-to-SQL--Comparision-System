import streamlit as st
import pandas as pd
import time
import warnings

from vanna_agent import get_vanna
from app import ask, clean_sql
from metrics_compare_methods import context_engineering_sql, react_sql, run_query

warnings.filterwarnings("ignore")


# scoring
def compute_score(baseline_df, test_result):
    if isinstance(test_result, str):
        return 0
    if not isinstance(test_result, pd.DataFrame):
        return 0
    if baseline_df is None or baseline_df.empty:
        return 1 if test_result.empty else 0.5
    try:
        if test_result.shape == baseline_df.shape and test_result.equals(baseline_df):
            return 1
        return 0.5
    except:
        return 0.5


# UI
st.set_page_config(page_title="NL2SQL Comparison", layout="wide")
st.title("NL → SQL Prompting Comparison")

question = st.text_input("Enter your question:")

if st.button("Run Comparison"):

    vn = get_vanna()

    #  Vanna
    start = time.time()
    vanna_state = ask(question)
    vanna_time = time.time() - start

    #  Context
    start = time.time()
    ce_sql = context_engineering_sql(question, vn)
    ce_result = run_query(vn, ce_sql)
    ce_time = time.time() - start

    #  ReAct
    start = time.time()
    react_query = react_sql(question, vn)
    react_result = run_query(vn, react_query)
    react_time = time.time() - start

    # Layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Vanna")
        st.code(clean_sql(vanna_state.get("sql")), language="sql")
        st.dataframe(vanna_state.get("result"))

    with col2:
        st.subheader("Context")
        st.code(clean_sql(ce_sql), language="sql")
        st.dataframe(ce_result)

    with col3:
        st.subheader("ReAct")
        st.code(clean_sql(react_query), language="sql")
        st.dataframe(react_result)

    # scoring
    baseline_df = vanna_state.get("result")

    scores = {
        "Vanna": 1,
        "Context": compute_score(baseline_df, ce_result),
        "ReAct": compute_score(baseline_df, react_result)
    }

    best = max(scores, key=scores.get)

    st.markdown("---")
    st.subheader("Accuracy Scores")
    st.write(scores)
    st.success(f"Best Method: {best}")

    # chart
    st.bar_chart(pd.DataFrame(scores.items(), columns=["Method", "Score"]).set_index("Method"))

    # timing
    times = {
        "Vanna": vanna_time,
        "Context": ce_time,
        "ReAct": react_time
    }

    st.markdown("---")
    st.subheader("Response Time")
    st.write(times)

    st.bar_chart(pd.DataFrame(times.items(), columns=["Method", "Time"]).set_index("Method"))

    fastest = min(times, key=times.get)
    st.success(f"⚡ Fastest Method: {fastest}")