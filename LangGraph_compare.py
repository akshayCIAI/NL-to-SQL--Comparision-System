import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyAk7gscNaKVRV80pvbDya-9qOK52XjQvQM"

import time
import warnings
import dspy
from dataclasses import dataclass, field
from typing import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from vanna_agent import get_vanna
from app import clean_sql

warnings.filterwarnings("ignore")


# =========================
# 🔧 DSPy SETUP
# =========================
lm = dspy.LM(
    model="gemini/gemini-flash-latest",
    api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.0,
    max_tokens=2000,
    cache=False,
)
dspy.configure(lm=lm)


# =========================
#  METRICS
# =========================
@dataclass
class Metrics:
    input_tokens:  int   = 0
    output_tokens: int   = 0
    total_tokens:  int   = 0
    reaction_ms:   float = 0.0

    def display(self):
        print(f"   Reaction time : {self.reaction_ms:.0f} ms")
        print(f"   Input tokens  : {self.input_tokens}")
        print(f"   Output tokens : {self.output_tokens}")
        print(f"   Total tokens  : {self.total_tokens}")


# =========================
# TOKEN HELPERS
# =========================
def extract_gemini_tokens(response) -> tuple[int, int]:
    try:
        usage = response.usage_metadata
        return usage.prompt_token_count, usage.candidates_token_count
    except Exception:
        return 0, 0


def extract_dspy_tokens() -> tuple[int, int]:
    try:
        history = dspy.settings.lm.history
        if not history:
            return 0, 0
        last = history[-1]

        response = last.get("response")
        if response:
            usage = getattr(response, "usage", None)
            if usage:
                return getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)

        usage = last.get("usage")
        if usage:
            if isinstance(usage, dict):
                return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            return getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)

        if response:
            meta = getattr(response, "usage_metadata", None)
            if meta:
                return getattr(meta, "prompt_token_count", 0), getattr(meta, "candidates_token_count", 0)
    except Exception:
        pass
    return 0, 0


# =========================
#  DSPy SIGNATURE
# =========================
class SQLTask(dspy.Signature):
    """Generate SQL from natural language"""
    question = dspy.InputField()
    schema   = dspy.InputField()
    sql      = dspy.OutputField()


class DSPyContext(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(SQLTask)

    def forward(self, question, schema):
        return self.generate(question=question, schema=schema)


class DSPyReAct(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(SQLTask)

    def forward(self, question, schema):
        return self.generate(question=question, schema=schema)


# =========================
# RUN SQL
# =========================
def run_query(vn, sql: str):
    try:
        sql = clean_sql(sql)
        return vn.run_sql(sql)
    except Exception as e:
        return str(e)


# =========================
#  LANGGRAPH STATE
# =========================
class CompareState(TypedDict):
    question:          str
    schema:            str

    # Each technique stores: {sql, result, metrics}
    context_normal:    dict
    context_dspy:      dict
    react_normal:      dict
    react_dspy:        dict


# =========================
# NODE 1: Context Normal
# =========================
def node_context_normal(state: CompareState) -> CompareState:
    print("\n[LangGraph] Running: Context Engineering (Normal)...")
    vn = get_vanna()

    prompt = f"""
You are a SQLite expert.

Tables:
customers(customer_id, name, country, signup_date)
products(product_id, name, category, unit_price)
orders(order_id, customer_id, product_id, quantity, order_date)

Rules:
- Return ONLY SQL
- Do NOT invent tables
- Use LOWER() for string comparisons

Question:
{state["question"]}
"""
    start    = time.perf_counter()
    response = vn.chat_model.generate_content(prompt)
    elapsed  = (time.perf_counter() - start) * 1000

    sql      = response.text.strip()
    result   = run_query(vn, sql)
    inp, out = extract_gemini_tokens(response)

    return {
        **state,
        "context_normal": {
            "sql":     sql,
            "result":  result,
            "metrics": Metrics(
                input_tokens  = inp,
                output_tokens = out,
                total_tokens  = inp + out,
                reaction_ms   = elapsed,
            ),
        },
    }


# =========================
#  NODE 2: Context DSPy
# =========================
def node_context_dspy(state: CompareState) -> CompareState:
    print("[LangGraph] Running: Context Engineering (DSPy)...")
    vn = get_vanna()

    try:
        model    = DSPyContext()
        start    = time.perf_counter()
        response = model(question=state["question"], schema=state["schema"])
        elapsed  = (time.perf_counter() - start) * 1000

        sql      = response.sql.strip()
        sql      = sql.replace("```sql", "").replace("```", "").strip()
        result   = run_query(vn, sql)
        inp, out = extract_dspy_tokens()

        return {
            **state,
            "context_dspy": {
                "sql":     sql,
                "result":  result,
                "metrics": Metrics(
                    input_tokens  = inp,
                    output_tokens = out,
                    total_tokens  = inp + out,
                    reaction_ms   = elapsed,
                ),
            },
        }
    except Exception as e:
        return {
            **state,
            "context_dspy": {
                "sql":     "ERROR",
                "result":  str(e),
                "metrics": Metrics(),
            },
        }


# =========================
#  NODE 3: ReAct Normal
# =========================
def node_react_normal(state: CompareState) -> CompareState:
    print("[LangGraph] Running: ReAct (Normal)...")
    vn = get_vanna()

    prompt = f"""
You are a SQLite expert using ReAct reasoning.

STRICT RULES:
- Use ONLY schema
- Return ONLY SQL
- Use LOWER() for comparisons
- Always use DISTINCT to avoid duplicate rows

Schema:
customers(customer_id, name, country, signup_date)
products(product_id, name, category, unit_price)
orders(order_id, customer_id, product_id, quantity, order_date)

Question: {state["question"]}

Thought:
- Identify tables
- Identify joins
- Identify columns
- Check if DISTINCT is needed

Action:
Generate SQL
"""
    start    = time.perf_counter()
    response = vn.chat_model.generate_content(prompt)
    elapsed  = (time.perf_counter() - start) * 1000

    text     = response.text.strip()
    inp, out = extract_gemini_tokens(response)

    if "SELECT" in text:
        sql = "SELECT " + text.split("SELECT", 1)[1]
    else:
        sql = text

    result = run_query(vn, sql)

    return {
        **state,
        "react_normal": {
            "sql":     sql,
            "result":  result,
            "metrics": Metrics(
                input_tokens  = inp,
                output_tokens = out,
                total_tokens  = inp + out,
                reaction_ms   = elapsed,
            ),
        },
    }


# =========================
#  NODE 4: ReAct DSPy
# =========================
def node_react_dspy(state: CompareState) -> CompareState:
    print("[LangGraph] Running: ReAct (DSPy)...")
    vn = get_vanna()

    try:
        model    = DSPyReAct()
        start    = time.perf_counter()
        response = model(question=state["question"], schema=state["schema"])
        elapsed  = (time.perf_counter() - start) * 1000

        sql = None
        if hasattr(response, "sql") and response.sql:
            sql = response.sql.strip()
        elif isinstance(response, dict) and "sql" in response:
            sql = response["sql"].strip()
        else:
            text = str(response)
            if "SELECT" in text:
                sql = "SELECT " + text.split("SELECT", 1)[1]

        if not sql:
            raise ValueError("No SQL generated")

        sql      = sql.replace("```sql", "").replace("```", "").strip()
        result   = run_query(vn, sql)
        inp, out = extract_dspy_tokens()

        return {
            **state,
            "react_dspy": {
                "sql":     sql,
                "result":  result,
                "metrics": Metrics(
                    input_tokens  = inp,
                    output_tokens = out,
                    total_tokens  = inp + out,
                    reaction_ms   = elapsed,
                ),
            },
        }
    except Exception as e:
        return {
            **state,
            "react_dspy": {
                "sql":     "ERROR",
                "result":  str(e),
                "metrics": Metrics(),
            },
        }


# =========================
#  NODE 5: Summarise
# =========================
def node_summarise(state: CompareState) -> CompareState:
    """Final node — prints all results and summary table."""

    techniques = [
        ("Context Engineering (Normal)", state["context_normal"]),
        ("Context Engineering (DSPy)",   state["context_dspy"]),
        ("ReAct (Normal)",               state["react_normal"]),
        ("ReAct (DSPy)",                 state["react_dspy"]),
    ]

    print("\n" + "=" * 45)
    print(f"  QUESTION: {state['question']}")
    print("=" * 45)

    for name, data in techniques:
        print(f"\n--- {name} ---")
        print(data["sql"])
        print(data["result"])
        data["metrics"].display()

    # Summary table
    print("\n" + "=" * 45)
    print("   SUMMARY")
    print("=" * 45)
    print(f"{'Technique':<30} {'Time(ms)':>9} {'In Tok':>8} {'Out Tok':>8} {'Total':>8}")
    print("-" * 68)
    for name, data in techniques:
        m = data["metrics"]
        print(f"{name:<30} {m.reaction_ms:>8.0f}ms {m.input_tokens:>8} {m.output_tokens:>8} {m.total_tokens:>8}")
    print("=" * 45 + "\n")

    return state


# =========================
# BUILD LANGGRAPH
# =========================
def build_graph() -> StateGraph:
    """
    Build the LangGraph workflow.

    Flow:
    context_normal → context_dspy → react_normal → react_dspy → summarise → END
    """
    graph = StateGraph(CompareState)

    # Add nodes
    graph.add_node("context_normal", node_context_normal)
    graph.add_node("context_dspy",   node_context_dspy)
    graph.add_node("react_normal",   node_react_normal)
    graph.add_node("react_dspy",     node_react_dspy)
    graph.add_node("summarise",      node_summarise)

    # Add edges (sequential flow)
    graph.set_entry_point("context_normal")
    graph.add_edge("context_normal", "context_dspy")
    graph.add_edge("context_dspy",   "react_normal")
    graph.add_edge("react_normal",   "react_dspy")
    graph.add_edge("react_dspy",     "summarise")
    graph.add_edge("summarise",      END)

    return graph.compile()


# =========================
# COMPARE via LangGraph
# =========================
def compare(question: str):
    schema = """
customers(customer_id, name, country, signup_date)
products(product_id, name, category, unit_price)
orders(order_id, customer_id, product_id, quantity, order_date)
"""
    # Initial state
    initial_state: CompareState = {
        "question":      question,
        "schema":        schema,
        "context_normal": {},
        "context_dspy":  {},
        "react_normal":  {},
        "react_dspy":    {},
    }

    # Build and run graph
    graph = build_graph()
    print("\n[LangGraph] Starting workflow...\n")
    final_state = graph.invoke(initial_state)
    return final_state


# =========================
#  RUN
# =========================
if __name__ == "__main__":
    q = input("Enter your question: ")
    compare(q)