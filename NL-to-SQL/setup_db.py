"""Create a small sample SQLite database for the demo."""
import os
import sqlite3

DB_PATH = "sample.db"

def create_sample_db(path: str = DB_PATH) -> None:
    if os.path.exists(path):
        os.remove(path)

    conn = sqlite3.connect(path)
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        country TEXT NOT NULL,
        signup_date DATE NOT NULL
    );

    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        unit_price REAL NOT NULL
    );

    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
        product_id INTEGER NOT NULL REFERENCES products(product_id),
        quantity INTEGER NOT NULL,
        order_date DATE NOT NULL
    );
    """)

    cur.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?)",
        [
            (1, "Alice", "USA", "2024-01-15"),
            (2, "Bob", "UK", "2024-02-10"),
            (3, "Charlie", "USA", "2024-03-05"),
            (4, "Diana", "India", "2024-04-20"),
            (5, "Ethan", "India", "2024-05-01"),
        ],
    )

    cur.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?)",
        [
            (1, "Laptop", "Electronics", 1200.00),
            (2, "Headphones", "Electronics", 150.00),
            (3, "Coffee Mug", "Kitchen", 12.50),
            (4, "Desk Chair", "Furniture", 300.00),
            (5, "Notebook", "Stationery", 5.00),
        ],
    )

    cur.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
        [
            (1, 1, 1, 1, "2024-06-01"),
            (2, 1, 2, 2, "2024-06-03"),
            (3, 2, 1, 1, "2024-06-05"),
            (4, 3, 1, 1, "2024-06-10"),
            (5, 3, 4, 1, "2024-06-12"),
            (6, 4, 5, 10, "2024-06-15"),
            (7, 5, 2, 2, "2024-06-18"),
            (8, 5, 3, 4, "2024-06-20"),
            (9, 2, 1, 1, "2024-06-22"),
            (10, 4, 4, 1, "2024-06-25"),
        ],
    )

    conn.commit()
    conn.close()
    print(f"Created sample database at {path}")


if __name__ == "__main__":
    create_sample_db()