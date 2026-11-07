from __future__ import annotations

from typing import Dict

import pandas as pd


class BusinessAssistant:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def answer(self, question: str) -> str:
        question = question.lower()
        if "revenue" in question or "sales" in question:
            if "country" in question:
                if "Country" in self.df.columns:
                    revenue_by_country = self.df.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
                    top = revenue_by_country.idxmax()
                    return f"The top revenue-generating country is {top} with {revenue_by_country.max():,.2f} in revenue."
            total_revenue = self.df["Revenue"].sum() if "Revenue" in self.df.columns else 0
            return f"Total revenue is {total_revenue:,.2f}."
        if "customer" in question:
            if "CustomerID" in self.df.columns:
                customers = self.df["CustomerID"].nunique()
                return f"There are {customers} unique customers in the dataset."
        if "product" in question:
            if "Description" in self.df.columns:
                top_product = self.df.groupby("Description")["Revenue"].sum().idxmax() if "Revenue" in self.df.columns else self.df["Description"].mode().iloc[0]
                return f"The highest-revenue product is {top_product}."
        if "anomaly" in question or "outlier" in question:
            return "An anomaly indicates an unusual transaction pattern and should be reviewed by business teams before any fraud conclusion is made."
        return "I can help summarize revenue, customer counts, top products, and anomaly trends in this business dataset."
