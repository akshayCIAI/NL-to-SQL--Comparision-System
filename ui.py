import streamlit as st
import pandas as pd
from LangGraph_compare import compare

st.set_page_config(page_title="NL → SQL Comparison", layout="wide")

st.title(" NL to SQL – Prompting Comparison Dashboard")

# =========================
# INPUT
# =========================
question = st.text_input("Enter your question:", placeholder="e.g. Which product performs best?")

if st.button("Run Comparison") and question:
    with st.spinner("Running all techniques..."):
        state = compare(question)

    st.success("Execution completed!")

    # Extract results
    methods = {
        "Context Normal": state["context_normal"],
        "Context DSPy": state["context_dspy"],
        "ReAct Normal": state["react_normal"],
        "ReAct DSPy": state["react_dspy"],
    }

    # =========================
    # SIDE BY SIDE DISPLAY
    # =========================
    st.subheader(" Results Comparison")

    cols = st.columns(4)

    for col, (name, data) in zip(cols, methods.items()):
        with col:
            st.markdown(f"### {name}")

            st.markdown("**SQL**")
            st.code(data["sql"], language="sql")

            st.markdown("**Result**")
            st.dataframe(data["result"], use_container_width=True)

            m = data["metrics"]

            st.markdown("**Metrics**")
            st.write(f" Time: {m.reaction_ms:.0f} ms")
            st.write(f" Input Tokens: {m.input_tokens}")
            st.write(f"Output Tokens: {m.output_tokens}")
            st.write(f" Total Tokens: {m.total_tokens}")

    # =========================
    # SUMMARY TABLE
    # =========================
    st.subheader(" Summary Table")

    summary_data = []
    for name, data in methods.items():
        m = data["metrics"]
        summary_data.append({
            "Technique": name,
            "Time (ms)": int(m.reaction_ms),
            "Input Tokens": m.input_tokens,
            "Output Tokens": m.output_tokens,
            "Total Tokens": m.total_tokens
        })

    df = pd.DataFrame(summary_data)

    st.dataframe(df, use_container_width=True)

    # =========================
    # BAR CHARTS (optional but nice)
    # =========================
    st.subheader(" Performance Charts")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("###  Response Time")
        st.bar_chart(df.set_index("Technique")["Time (ms)"])

    with col2:
        st.markdown("### Token Usage")
        st.bar_chart(df.set_index("Technique")["Total Tokens"])