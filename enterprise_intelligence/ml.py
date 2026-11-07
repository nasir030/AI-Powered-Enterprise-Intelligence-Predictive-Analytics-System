from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


class SalesForecastModel:
    def __init__(self, horizon: int = 30) -> None:
        self.horizon = horizon

    def _prepare_series(self, series: pd.Series, frequency: str = "daily") -> pd.DataFrame:
        if not isinstance(series, pd.Series):
            raise TypeError("Expected a Pandas Series for forecasting.")
        values = series.copy().sort_index()
        values = values.asfreq("D") if frequency.lower() in {"daily", "d"} else values
        if values.index.freq is None:
            values = values.asfreq("D")
        df = pd.DataFrame({"sales": values})
        df["date"] = df.index
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df["lag_1"] = df["sales"].shift(1)
        df["lag_7"] = df["sales"].shift(7)
        return df.dropna().reset_index(drop=True)

    def forecast(self, series: pd.Series, horizon: int = 30, frequency: str = "daily") -> Dict[str, object]:
        horizon = int(horizon or self.horizon)
        prepared = self._prepare_series(series, frequency)
        if len(prepared) < 10:
            baseline = series.tail(1).iloc[0] if not series.empty else 0
            forecast_values = [baseline for _ in range(horizon)]
            future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
            return {
                "forecast": pd.Series(forecast_values, index=future_dates),
                "metrics": {"mae": 0.0, "rmse": 0.0, "mape": 0.0},
                "model_name": "moving_average",
            }

        train_size = max(1, int(len(prepared) * 0.8))
        train = prepared.iloc[:train_size]
        test = prepared.iloc[train_size:]

        features = ["day_of_week", "day_of_month", "month", "year", "lag_1", "lag_7"]
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(train[features], train["sales"])

        pred = model.predict(test[features])
        actual = test["sales"].to_numpy()
        mae = float(mean_absolute_error(actual, pred))
        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        mape = float(np.mean(np.abs((actual - pred) / np.maximum(actual, 1e-8))) * 100)

        last_date = prepared["date"].iloc[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        future_frame = pd.DataFrame({
            "date": future_dates,
            "day_of_week": future_dates.dayofweek,
            "day_of_month": future_dates.day,
            "month": future_dates.month,
            "year": future_dates.year,
        })
        future_frame["lag_1"] = series.iloc[-1] if isinstance(series.iloc[-1], (int, float, np.number)) else 0
        future_frame["lag_7"] = series.iloc[-7] if len(series) >= 7 else series.iloc[-1]

        future_pred = model.predict(future_frame[features])
        forecast = pd.Series(future_pred, index=future_dates, name="forecast")
        return {
            "forecast": forecast,
            "metrics": {"mae": mae, "rmse": rmse, "mape": mape},
            "model_name": "random_forest",
        }


class CustomerSegmentationModel:
    def __init__(self, n_clusters: int = 3) -> None:
        self.n_clusters = n_clusters

    def _build_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        if "InvoiceDate" not in frame.columns:
            raise ValueError("InvoiceDate column required for RFM analysis.")
        if "CustomerID" not in frame.columns:
            raise ValueError("CustomerID column required for RFM analysis.")
        if "Revenue" not in frame.columns:
            frame["Revenue"] = 0
        frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])
        snapshot_date = frame["InvoiceDate"].max() + pd.Timedelta(days=1)
        rfm = frame.groupby("CustomerID").agg(
            recency=("InvoiceDate", lambda s: (snapshot_date - s.max()).days),
            frequency=("InvoiceDate", "count"),
            monetary=("Revenue", "sum"),
        ).reset_index()
        return rfm

    def segment(self, df: pd.DataFrame, n_clusters: int = None) -> Dict[str, object]:
        n_clusters = n_clusters or self.n_clusters
        rfm = self._build_rfm(df)
        features = rfm[["recency", "frequency", "monetary"]].copy()
        scaler = features.std(ddof=0).replace(0, 1)
        scaled = (features - features.mean()) / scaler
        cluster_count = min(max(2, n_clusters), len(rfm))
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        rfm["segment"] = labels
        summary = rfm.groupby("segment").agg(
            customers=("CustomerID", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        return {"segments": rfm, "summary": summary, "cluster_model": model}


class AnomalyDetector:
    def __init__(self, contamination: float = 0.05) -> None:
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def detect(self, df: pd.DataFrame, numeric_columns: List[str] | None = None) -> Dict[str, object]:
        frame = df.copy()
        if numeric_columns is None:
            numeric_columns = ["Quantity", "UnitPrice", "Revenue"]
        available = [col for col in numeric_columns if col in frame.columns]
        if not available:
            raise ValueError("No numeric columns available for anomaly detection.")
        model_input = frame[available].fillna(0)
        labels = self.model.fit_predict(model_input)
        frame["is_anomaly"] = labels == -1
        frame["anomaly_score"] = -self.model.score_samples(model_input)
        anomalies = frame[frame["is_anomaly"]]
        return {"full_data": frame, "anomalies": anomalies, "model": self.model}


class ProductRecommendationEngine:
    def __init__(self, top_n: int = 5) -> None:
        self.top_n = top_n

    def recommend(self, df: pd.DataFrame, customer_id_col: str = "CustomerID", product_col: str = "Description") -> Dict[str, List[str]]:
        if customer_id_col not in df.columns or product_col not in df.columns:
            raise ValueError("CustomerID and Description columns are required.")
        purchases = df.groupby([customer_id_col, product_col]).size().reset_index(name="count")
        recommendations: Dict[str, List[str]] = {}
        for customer_id, group in purchases.groupby(customer_id_col):
            items = group.sort_values("count", ascending=False)[product_col].head(self.top_n).tolist()
            recommendations[str(customer_id)] = items
        return recommendations
