from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = {
    "invoice": "InvoiceNo",
    "invoice_no": "InvoiceNo",
    "invoice number": "InvoiceNo",
    "invoiceid": "InvoiceNo",
    "invoice_id": "InvoiceNo",
    "invoiceno": "InvoiceNo",
    "date": "InvoiceDate",
    "invoice date": "InvoiceDate",
    "datetime": "InvoiceDate",
    "order date": "InvoiceDate",
    "transaction date": "InvoiceDate",
    "customer": "CustomerID",
    "customer id": "CustomerID",
    "customerid": "CustomerID",
    "customer_id": "CustomerID",
    "stockcode": "StockCode",
    "product": "Description",
    "product name": "Description",
    "description": "Description",
    "product description": "Description",
    "quantity": "Quantity",
    "qty": "Quantity",
    "unit price": "UnitPrice",
    "unitprice": "UnitPrice",
    "price": "UnitPrice",
    "country": "Country",
    "revenue": "Revenue",
    "sales": "Revenue",
    "total": "Revenue",
    "amount": "Revenue",
}


def standardize_column_names(columns: Iterable[str]) -> List[str]:
    cleaned = []
    for col in columns:
        name = str(col).strip()
        name = re.sub(r"[^a-zA-Z0-9]+", " ", name)
        name = " ".join(name.lower().split())
        cleaned.append(name)
    return cleaned


def _normalized_lookup_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def infer_mapping(df: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    alias_map = { _normalized_lookup_key(key): value for key, value in CANONICAL_COLUMNS.items() }
    for original in df.columns:
        lookup_key = _normalized_lookup_key(original)
        canonical = alias_map.get(lookup_key)
        if canonical is not None:
            mapping[original] = canonical
    return mapping


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _coerce_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    return parsed


def _safe_value_counts(series: pd.Series) -> pd.Series:
    return series.dropna().astype(str).str.strip()


def generate_demo_dataframe() -> pd.DataFrame:
    rng = pd.date_range("2024-01-01", periods=180, freq="D")
    rows = []
    products = ["Laptop", "Mouse", "Keyboard", "Monitor", "Desk Chair", "Printer", "USB Cable"]
    countries = ["United States", "United Kingdom", "Germany", "France", "India"]
    for idx, date in enumerate(rng):
        for customer_id in range(1, 11):
            if idx % 7 == customer_id % 7:
                product = products[(idx + customer_id) % len(products)]
                quantity = 1 + ((customer_id + idx) % 4)
                unit_price = 50 + ((customer_id * 7 + idx) % 120)
                revenue = round(quantity * unit_price, 2)
                row = {
                    "InvoiceNo": f"INV-{idx + 1}-{customer_id}",
                    "CustomerID": customer_id,
                    "InvoiceDate": date,
                    "Description": product,
                    "Quantity": quantity,
                    "UnitPrice": unit_price,
                    "Country": countries[(customer_id + idx) % len(countries)],
                    "Revenue": revenue,
                }
                rows.append(row)
    return pd.DataFrame(rows)


def clean_business_data(df: pd.DataFrame, mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Input data is empty.")

    df = df.copy()
    df.columns = standardize_column_names(df.columns)

    if mapping is None:
        mapping = infer_mapping(df)

    renamed_columns = {col: target for col, target in mapping.items() if col in df.columns}
    df = df.rename(columns=renamed_columns)

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = _coerce_datetime(df["InvoiceDate"])
    if "Quantity" in df.columns:
        df["Quantity"] = _coerce_numeric(df["Quantity"]) 
    if "UnitPrice" in df.columns:
        df["UnitPrice"] = _coerce_numeric(df["UnitPrice"])
    if "Revenue" not in df.columns and "Quantity" in df.columns and "UnitPrice" in df.columns:
        df["Revenue"] = (df["Quantity"] * df["UnitPrice"]).fillna(0)

    if "CustomerID" in df.columns:
        df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")

    initial_rows = len(df)
    if "Quantity" in df.columns:
        df = df[(df["Quantity"].notna()) & (df["Quantity"] > 0)]
    if "UnitPrice" in df.columns:
        df = df[(df["UnitPrice"].notna()) & (df["UnitPrice"] > 0)]
    if "Revenue" in df.columns:
        df["Revenue"] = _coerce_numeric(df["Revenue"]).fillna(0)

    df = df.dropna(subset=[col for col in ["InvoiceNo", "InvoiceDate"] if col in df.columns], how="any")
    df = df.drop_duplicates().reset_index(drop=True)

    if "InvoiceNo" in df.columns:
        df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    if "Description" in df.columns:
        df["Description"] = df["Description"].fillna("Unknown Product").astype(str).str.strip()
    if "Country" in df.columns:
        df["Country"] = df["Country"].fillna("Unknown").astype(str).str.strip()

    return df


def summarize_data_quality(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, cleaning_operations: Optional[List[str]] = None) -> Dict[str, Any]:
    original_df = original_df.copy()
    cleaned_df = cleaned_df.copy()
    ops = cleaning_operations or []

    total_rows = len(original_df)
    total_columns = len(original_df.columns)
    missing_values = int(original_df.isna().sum().sum())
    duplicate_rows = int(original_df.duplicated().sum())
    invalid_records = max(0, total_rows - len(cleaned_df))
    dtypes = original_df.dtypes.astype(str).to_dict()
    data_dict = {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "invalid_records": invalid_records,
        "data_types": dtypes,
        "cleaning_summary": ops,
        "rows_after_cleaning": len(cleaned_df),
        "cleaned_columns": list(cleaned_df.columns),
        "profit_data_available": bool("Profit" in cleaned_df.columns or "Cost" in cleaned_df.columns or "ProfitMargin" in cleaned_df.columns),
    }
    return data_dict


def process_uploaded_file(file_obj: Any) -> Dict[str, Any]:
    filename = getattr(file_obj, "name", "upload.csv")
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(file_obj)
    elif filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_obj)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")

    original = df.copy()
    cleaned = clean_business_data(df)
    operations = [
        "Standardized column names.",
        "Converted date columns to datetime.",
        "Removed rows with invalid quantity or price.",
        "Dropped duplicate rows.",
        "Filled missing product or country labels where possible.",
    ]
    report = summarize_data_quality(original, cleaned, operations)
    return {"data": cleaned, "report": report}
