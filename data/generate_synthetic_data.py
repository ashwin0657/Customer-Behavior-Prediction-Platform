import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
N_CUSTOMERS = 1200
N_PRODUCTS = 250
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)
COUNTRIES = ["United Kingdom", "Germany", "France", "India", "Spain", "USA", "Netherlands"]

customer_ids = np.arange(10000, 10000 + N_CUSTOMERS)
customer_type = np.random.choice(["loyal", "occasional", "one_time", "lapsing"], size=N_CUSTOMERS, p=[0.2, 0.35, 0.25, 0.2])
customer_country = np.random.choice(COUNTRIES, size=N_CUSTOMERS)
products = [f"PROD{str(i).zfill(4)}" for i in range(N_PRODUCTS)]
descriptions = [f"Product {i} - Item" for i in range(N_PRODUCTS)]
prices = np.round(np.random.uniform(1.5, 80, N_PRODUCTS), 2)

rows = []
invoice_counter = 500000
for idx, cust in enumerate(customer_ids):
    ctype = customer_type[idx]
    if ctype == "loyal":
        n_orders = np.random.randint(15, 40); span_days = np.random.randint(500, 730)
    elif ctype == "occasional":
        n_orders = np.random.randint(4, 14); span_days = np.random.randint(300, 600)
    elif ctype == "one_time":
        n_orders = 1; span_days = 1
    else:
        n_orders = np.random.randint(3, 10); span_days = np.random.randint(60, 200)

    if ctype == "lapsing":
        order_start = START_DATE
        order_days = sorted(np.random.randint(0, span_days, n_orders))
    else:
        max_start = (END_DATE - START_DATE).days - span_days
        offset = np.random.randint(0, max(max_start, 1))
        order_start = START_DATE + timedelta(days=int(offset))
        order_days = sorted(np.random.randint(0, span_days, n_orders))

    for od in order_days:
        invoice_counter += 1
        invoice_date = order_start + timedelta(days=int(od), hours=np.random.randint(8, 20))
        n_items = np.random.randint(1, 8)
        item_idxs = np.random.choice(N_PRODUCTS, n_items, replace=False)
        for pi in item_idxs:
            qty = np.random.randint(1, 12)
            if np.random.rand() < 0.03:
                qty = -abs(qty)
            rows.append({
                "InvoiceNo": invoice_counter, "StockCode": products[pi], "Description": descriptions[pi],
                "Quantity": qty, "InvoiceDate": invoice_date, "UnitPrice": prices[pi],
                "CustomerID": cust, "Country": customer_country[idx],
            })

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/raw_transactions.csv", index=False)
print(f"Generated {len(df):,} rows for {N_CUSTOMERS} customers -> data/raw_transactions.csv")
