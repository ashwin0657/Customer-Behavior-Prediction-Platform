import pandas as pd


def load_raw(path="data/raw_transactions.csv"):
    return pd.read_csv(path, parse_dates=["InvoiceDate"])


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(int)

    junk_codes = ["POST", "DOT", "BANK CHARGES", "AMAZONFEE", "M"]
    df = df[~df["StockCode"].astype(str).str.upper().isin(junk_codes)]

    df["IsReturn"] = df["Quantity"] < 0
    df = df[df["UnitPrice"] > 0]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    after = len(df)
    print(f"Cleaned transactions: {before:,} -> {after:,} rows "
          f"({before - after:,} removed, {df['IsReturn'].sum():,} flagged as returns)")
    return df


if __name__ == "__main__":
    raw = load_raw()
    clean = clean_transactions(raw)
    clean.to_csv("data/clean_transactions.csv", index=False)
    print("Saved -> data/clean_transactions.csv")
    print(clean.head())
