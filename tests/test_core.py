import pandas as pd

from enterprise_intelligence.analytics import compute_kpis
from enterprise_intelligence.data_pipeline import clean_business_data
from enterprise_intelligence.ml import CustomerSegmentationModel, SalesForecastModel


def test_clean_business_data_handles_bad_rows():
    df = pd.DataFrame(
        {
            "Invoice": ["A1", "A2", "A3"],
            "StockCode": ["100", "200", "300"],
            "Description": ["Widget", "Thing", "Other"],
            "Quantity": [5, -2, 10],
            "UnitPrice": [10.5, 0, 15.75],
            "CustomerID": [1, 2, None],
            "Country": ["US", "UK", "US"],
            "InvoiceDate": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )

    cleaned = clean_business_data(df)
    assert len(cleaned) == 2
    assert cleaned["Revenue"].sum() > 0
    assert "InvoiceDate" in cleaned.columns


def test_compute_kpis_returns_expected_metrics():
    df = pd.DataFrame(
        {
            "InvoiceNo": ["A1", "A2"],
            "CustomerID": [1, 2],
            "InvoiceDate": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "Quantity": [2, 3],
            "UnitPrice": [10, 20],
            "Revenue": [20, 60],
            "Description": ["X", "Y"],
        }
    )

    metrics = compute_kpis(df)
    assert metrics["total_revenue"] == 80
    assert metrics["total_orders"] == 2
    assert metrics["total_customers"] == 2


def test_forecast_model_creates_prediction():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    sales = pd.Series(range(1, 121), index=dates, name="sales")
    forecast = SalesForecastModel().forecast(sales, horizon=7, frequency="daily")
    assert len(forecast["forecast"]) == 7
    assert "rmse" in forecast["metrics"]


def test_customer_segmentation_returns_segments():
    df = pd.DataFrame(
        {
            "CustomerID": [1, 1, 2, 2, 3, 3],
            "InvoiceDate": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-03", "2024-01-05", "2024-01-02", "2024-01-04"]),
            "Revenue": [50, 100, 30, 30, 10, 20],
        }
    )

    model = CustomerSegmentationModel()
    result = model.segment(df)
    assert "segments" in result
    assert not result["segments"].empty
