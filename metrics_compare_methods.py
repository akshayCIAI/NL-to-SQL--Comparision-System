import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyAk7gscNaKVRV80pvbDya-9qOK52XjQvQM"

import time
import dspy
import warnings
from dataclasses import dataclass
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
# 📊 METRICS DATACLASS
# =========================
@dataclass
class Metrics:
    input_tokens:  int   = 0
    output_tokens: int   = 0
    total_tokens:  int   = 0
    reaction_ms:   float = 0.0

    def display(self):
        print(f"  ⏱  Reaction time : {self.reaction_ms:.0f} ms")
        print(f"  🔢 Input tokens  : {self.input_tokens}")
        print(f"  🔢 Output tokens : {self.output_tokens}")
        print(f"  🔢 Total tokens  : {self.total_tokens}")


# =========================
# 🔢 TOKEN HELPERS
# =========================
def extract_gemini_tokens(response) -> tuple[int, int]:
    """Extract token counts from a Gemini API response object."""
    try:
        usage = response.usage_metadata
        return usage.prompt_token_count, usage.candidates_token_count
    except Exception:
        pass
    try:
        # Fallback: some response objects use different attribute names
        return response.prompt_token_count, response.candidates_token_count
    except Exception:
        return 0, 0


def extract_dspy_tokens() -> tuple[int, int]:
    """Try all known DSPy/LiteLLM history paths to get token counts."""
    try:
        history = dspy.settings.lm.history
        if not history:
            return 0, 0

        last = history[-1]

        # Path 1: LiteLLM response object (DSPy 2.4+)
        response = last.get("response")
        if response:
            usage = getattr(response, "usage", None)
            if usage:
                return getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)

        # Path 2: usage directly on last entry
        usage = last.get("usage")
        if usage:
            if isinstance(usage, dict):
                return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            return getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)

        # Path 3: outputs list
        outputs = last.get("outputs", [])
        if outputs:
            usage = getattr(outputs[0], "usage", None)
            if usage:
                return getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)

        # Path 4: nested under response.usage_metadata (Gemini specific)
        if response:
            meta = getattr(response, "usage_metadata", None)
            if meta:
                return getattr(meta, "prompt_token_count", 0), getattr(meta, "candidates_token_count", 0)

    except Exception as e:
        print(f"[Token extract error]: {e}")

    return 0, 0

# =========================
# 🧠 DSPy SIGNATURE
# =========================
class SQLTask(dspy.Signature):
    """Generate SQL from natural language"""
    question = dspy.InputField()
    schema   = dspy.InputField()
    sql      = dspy.OutputField()


# =========================
# 🧠 DSPy MODULES
# =========================
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
# 🧪 RUN SQL
# =========================
def run_query(vn, sql):
    try:
        sql = clean_sql(sql)
        return vn.run_sql(sql)
    except Exception as e:
        return str(e)


# =========================
# 🟡 CONTEXT (NORMAL)
# =========================
def context_engineering_sql(question, vn) -> tuple[str, Metrics]:
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
{question}
"""
    start    = time.perf_counter()
    response = vn.chat_model.generate_content(prompt)
    elapsed  = (time.perf_counter() - start) * 1000

    inp, out = extract_gemini_tokens(response)
    metrics  = Metrics(
        input_tokens  = inp,
        output_tokens = out,
        total_tokens  = inp + out,
        reaction_ms   = elapsed,
    )
    return response.text.strip(), metrics


# =========================
# 🔵 REACT (NORMAL)
# =========================
def react_sql(question, vn) -> tuple[str, Metrics]:
    prompt = f"""
You are a SQLite expert using ReAct reasoning.

STRICT RULES:
- Use ONLY schema
- Return ONLY SQL
- Use LOWER() for comparisons

Schema:
customers(customer_id, name, country, signup_date)
products(product_id, name, category, unit_price)
orders(order_id, customer_id, product_id, quantity, order_date)

Question: {question}

Thought:
- Identify tables
- Identify joins
- Identify columns

Action:
Generate SQL
"""
    start    = time.perf_counter()
    response = vn.chat_model.generate_content(prompt)
    elapsed  = (time.perf_counter() - start) * 1000

    text     = response.text.strip()
    inp, out = extract_gemini_tokens(response)
    metrics  = Metrics(
        input_tokens  = inp,
        output_tokens = out,
        total_tokens  = inp + out,
        reaction_ms   = elapsed,
    )

    # clean output
    if "SELECT" in text:
        return "SELECT " + text.split("SELECT", 1)[1], metrics
    return text, metrics


# =========================
# 🟡 CONTEXT (DSPy)
# =========================
def run_context_dspy(question, vn) -> tuple[str, any, Metrics]:
    try:
        schema = """
customers(customer_id, name, country, signup_date)
products(product_id, name, category, unit_price)
orders(order_id, customer_id, product_id, quantity, order_date)
"""
        model   = DSPyContext()
        start   = time.perf_counter()
        response = model(question=question, schema=schema)
        elapsed = (time.perf_counter() - start) * 1000

        sql    = response.sql.strip()
        result = run_query(vn, sql)

        inp, out = extract_dspy_tokens()
        metrics  = Metrics(
            input_tokens  = inp,
            output_tokens = out,
            total_tokens  = inp + out,
            reaction_ms   = elapsed,
        )
        return sql, result, metrics

    except Exception as e:
        return "ERROR", str(e), Metrics()


# =========================
# 🔵 REACT (DSPy)
# =========================
def run_react_dspy(question, vn) -> tuple[str, any, Metrics]:
    try:
        schema = """
customers(customer_id, name, country, signup_date)
products(product_id, name, category, unit_price)
orders(order_id, customer_id, product_id, quantity, order_date)
"""
        model   = DSPyReAct()
        start   = time.perf_counter()
        response = model(question=question, schema=schema)
        elapsed  = (time.perf_counter() - start) * 1000

        print("\n[DEBUG DSPy RAW]:", response)

        sql = None
        if response:
            if hasattr(response, "sql") and response.sql:
                sql = response.sql.strip()
            elif isinstance(response, dict) and "sql" in response:
                sql = response["sql"].strip()
            else:
                text = str(response)
                if "SELECT" in text:
                    sql = "SELECT " + text.split("SELECT", 1)[1]

        if not sql:
            return "ERROR", "No SQL generated by DSPy", Metrics(reaction_ms=elapsed)

        sql    = sql.replace("```sql", "").replace("```", "").strip()
        result = run_query(vn, sql)

        inp, out = extract_dspy_tokens()
        metrics  = Metrics(
            input_tokens  = inp,
            output_tokens = out,
            total_tokens  = inp + out,
            reaction_ms   = elapsed,
        )
        return sql, result, metrics

    except Exception as e:
        return "ERROR", str(e), Metrics()


# =========================
# 🎯 MAIN COMPARISON
# =========================
def compare(question: str):
    vn = get_vanna()

    # ── Run all 4 techniques ──────────────────────────────────────────────
    ce_sql,        ce_metrics        = context_engineering_sql(question, vn)
    ce_result                        = run_query(vn, ce_sql)

    react_q,       react_metrics     = react_sql(question, vn)
    react_result                     = run_query(vn, react_q)

    ce_dspy_sql,   ce_dspy_result,   ce_dspy_metrics   = run_context_dspy(question, vn)
    react_dspy_sql,react_dspy_result,react_dspy_metrics = run_react_dspy(question, vn)

    # ── Display ───────────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print(f"  QUESTION: {question}")
    print("=" * 45)

    print("\n--- Context Engineering (Normal) ---")
    print(ce_sql)
    print(ce_result)
    ce_metrics.display()

    print("\n--- Context Engineering (DSPy) ---")
    print(ce_dspy_sql)
    print(ce_dspy_result)
    ce_dspy_metrics.display()

    print("\n--- ReAct (Normal) ---")
    print(react_q)
    print(react_result)
    react_metrics.display()

    print("\n--- ReAct (DSPy) ---")
    print(react_dspy_sql)
    print(react_dspy_result)
    react_dspy_metrics.display()

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print("  📊 SUMMARY")
    print("=" * 45)
    print(f"{'Technique':<30} {'Time(ms)':>8} {'In Tok':>8} {'Out Tok':>8} {'Total':>8}")
    print("-" * 65)
    rows = [
        ("Context Engineering (Normal)", ce_metrics),
        ("Context Engineering (DSPy)",   ce_dspy_metrics),
        ("ReAct (Normal)",               react_metrics),
        ("ReAct (DSPy)",                 react_dspy_metrics),
    ]
    for name, m in rows:
        print(f"{name:<30} {m.reaction_ms:>7.0f}ms {m.input_tokens:>8} {m.output_tokens:>8} {m.total_tokens:>8}")
    print("=" * 45 + "\n")


# =========================
# ▶️ RUN
# =========================
if __name__ == "__main__":
    q = input("Enter your question: ")
    compare(q)