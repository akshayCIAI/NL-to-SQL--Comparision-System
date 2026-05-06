from vanna.chromadb import ChromaDB_VectorStore
from vanna.google import GoogleGeminiChat

import os
from dotenv import load_dotenv

load_dotenv()

class MyVanna(ChromaDB_VectorStore, GoogleGeminiChat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        GoogleGeminiChat.__init__(self, config=config)

def train(vn: MyVanna) -> None:
    ddl_statements = [
        """
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            country TEXT NOT NULL,
            signup_date DATE NOT NULL
        );
        """,
        """
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            unit_price REAL NOT NULL
        );
        """,
        """
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
            product_id INTEGER NOT NULL REFERENCES products(product_id),
            quantity INTEGER NOT NULL,
            order_date DATE NOT NULL
        );
        """
    ]

    for ddl in ddl_statements:
        vn.train(ddl=ddl)

    vn.train(documentation="""
    Generate SQL based on schema.
    Do NOT reuse previous queries unless the question is exactly identical.
    Return only SQL.
    """)
    vn.train(
        question="How many customers do we have per country?",
        sql="SELECT country, COUNT(*) AS num_customers FROM customers GROUP BY country"
    )

    vn.train(
        question="What is the total revenue per product?",
        sql="""
        SELECT p.name,
               SUM(o.quantity * p.unit_price) AS revenue
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        GROUP BY p.name
        ORDER BY revenue DESC;
        """
    )
def get_vanna():
    vn = MyVanna(config={
        "model_name": "gemini-flash-lite-latest",
        "api_key": os.getenv("GOOGLE_API_KEY"),
    })

    vn.connect_to_sqlite("sample.db")
    train(vn)

    return vn