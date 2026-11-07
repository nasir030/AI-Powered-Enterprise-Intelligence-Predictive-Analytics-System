from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    df = df.copy()
    if df.empty:
        return {
            "total_revenue": 0.0,
            "total_orders": 0,
            "total_units_sold": 0,
            "average_order_value": 0.0,
            "total_customers": 0,
            "total_products": 0,
            "average_selling_price": 0.0,
            "profit": None,
            "profit_margin": None,
        }

    if "Revenue" not in df.columns:
        if {"Quantity", "UnitPrice"}.issubset(df.columns):
            df["Revenue"] = df["Quantity"] * df["UnitPrice"]
        else:
            df["Revenue"] = 0

    if "InvoiceNo" in df.columns:
        total_orders = df["InvoiceNo"].nunique()
    else:
        total_orders = len(df)

    total_revenue = float(df["Revenue"].sum())
    total_units_sold = int(df["Quantity"].sum()) if "Quantity" in df.columns else 0
    average_order_value = total_revenue / total_orders if total_orders else 0.0
    total_customers = int(df["CustomerID"].nunique()) if "CustomerID" in df.columns else 0
    total_products = int(df["Description"].nunique()) if "Description" in df.columns else 0
    average_selling_price = float(df["Revenue"].sum() / max(df["Quantity"].sum(), 1)) if "Quantity" in df.columns and df["Quantity"].sum() > 0 else 0.0

    profit = None
    profit_margin = None
    if "Profit" in df.columns:
        profit = float(df["Profit"].sum())
        if total_revenue:
            profit_margin = profit / total_revenue
    elif "Cost" in df.columns:
        profit = float((df["Revenue"] - df["Cost"]).sum())
        if total_revenue:
            profit_margin = profit / total_revenue

    return {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "total_units_sold": total_units_sold,
        "average_order_value": average_order_value,
        "total_customers": total_customers,
        "total_products": total_products,
        "average_selling_price": average_selling_price,
        "profit": profit,
        "profit_margin": profit_margin,
    }


def sales_time_series(df: pd.DataFrame, date_col: str = "InvoiceDate", value_col: str = "Revenue", freq: str = "M") -> pd.Series:
    frame = df.copy()
    if date_col not in frame.columns:
        raise ValueError(f"Column '{date_col}' not found in dataset.")
    if value_col not in frame.columns:
        frame[value_col] = 0
    frame = frame[[date_col, value_col]].dropna(subset=[date_col])
    frame[date_col] = pd.to_datetime(frame[date_col])
    series = frame.groupby(pd.Grouper(key=date_col, freq=freq))[value_col].sum()
    return series.sort_index()


def generate_business_insights(df: pd.DataFrame) -> Dict[str, List[dict]]:
    if df.empty:
        return {"top_products": [], "top_countries": [], "best_sales_period": [], "opportunities": []}

    insights: Dict[str, List[dict]] = {}

    if "Description" in df.columns and "Revenue" in df.columns:
        product_summary = df.groupby("Description")["Revenue"].sum().sort_values(ascending=False)
        insights["top_products"] = [
            {"product": product, "revenue": round(float(revenue), 2)}
            for product, revenue in product_summary.head(5).items()
        ]

    if "Country" in df.columns and "Revenue" in df.columns:
        country_summary = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
        insights["top_countries"] = [
            {"country": country, "revenue": round(float(revenue), 2)}
            for country, revenue in country_summary.head(5).items()
        ]

    if "InvoiceDate" in df.columns:
        date_summary = sales_time_series(df, freq="M")
        best_period = date_summary.idxmax() if not date_summary.empty else None
        if best_period is not None:
            insights["best_sales_period"] = [{"period": best_period.strftime("%Y-%m"), "revenue": round(float(date_summary.max()), 2)}]
        else:
            insights["best_sales_period"] = []

    opportunities = []
    if "Country" in df.columns and "Revenue" in df.columns:
        best_country = df.groupby("Country")["Revenue"].sum().idxmax() if not df.empty else None
        if best_country:
            opportunities.append({"title": "Growth opportunity", "detail": f"{best_country} is the strongest market and could be expanded with bundle offers."})
    if "Description" in df.columns and "Revenue" in df.columns:
        top_product = df.groupby("Description")["Revenue"].sum().idxmax() if not df.empty else None
        if top_product:
            opportunities.append({"title": "Promote winning products", "detail": f"{top_product} is driving the most revenue; focus marketing on this product line."})
    insights["opportunities"] = opportunities
    return insights
