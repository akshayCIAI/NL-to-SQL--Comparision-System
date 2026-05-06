from typing import Optional, TypedDict
import pandas as pd
from langgraph.graph import END, START, StateGraph

from vanna_agent import get_vanna
from dotenv import load_dotenv

import sys
import os
import logging

load_dotenv()

# Optional: reduce logging noise
logging.getLogger().setLevel(logging.ERROR)


class NL2SQLState(TypedDict, total=False):
    question: str
    sql: str
    result: Optional[pd.DataFrame]
    answer: Optional[str]
    error: Optional[str]


def generate_sql_node(state: NL2SQLState) -> NL2SQLState:
    vn = get_vanna()
    sql = vn.generate_sql(question=state["question"], allow_llm_to_see_data=True)
    return {"sql": sql}


def clean_sql(sql: str) -> str:
    if not sql:
        return sql

    # remove markdown blocks
    sql = sql.replace("```sqlite", "")
    sql = sql.replace("```sql", "")
    sql = sql.replace("```", "")

    # remove accidental prefix
    sql = sql.strip()
    if sql.lower().startswith("sqlite"):
        sql = sql[len("sqlite"):].strip()

    # keep only first SQL statement (prevents extra text errors)
    if ";" in sql:
        sql = sql.split(";")[0] + ";"

    return sql


def execute_sql_node(state):
    vn = get_vanna()

    try:
        clean_query = clean_sql(state["sql"])
        df = vn.run_sql(clean_query)

        return {"result": df, "sql": clean_query}

    except Exception as e:
        return {"error": str(e), "sql": state.get("sql")}


def summarize_node(state: NL2SQLState) -> NL2SQLState:
    if state.get("error"):
        return {"answer": f"Query failed: {state['error']}"}

    df = state.get("result")
    if df is None or df.empty:
        return {"answer": "The query returned no rows."}

    return {"answer": f"Returned {len(df)} row(s)."}


def build_graph():
    g = StateGraph(NL2SQLState)

    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute_sql", execute_sql_node)
    g.add_node("summarize", summarize_node)

    g.add_edge(START, "generate_sql")
    g.add_edge("generate_sql", "execute_sql")
    g.add_edge("execute_sql", "summarize")
    g.add_edge("summarize", END)

    return g.compile()


def ask(question: str):
    graph = build_graph()
    return graph.invoke({"question": question})


def main():
    from setup_db import create_sample_db

    create_sample_db()

    print("Vanna + LangGraph NL->SQL demo (type 'exit' to quit)\n")

    while True:
        try:
            q = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        # 🔥 suppress Vanna internal prints
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        try:
            state = ask(q)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        # clean output only
        print("\n--- SQL ---")
        print(state.get("sql"))

        df = state.get("result")
        if df is not None:
            print("\n--- Rows ---")
            print(df.to_string(index=False))

        print("\n--- Answer ---")
        print(state.get("answer"))
        print()


if __name__ == "__main__":
    main()