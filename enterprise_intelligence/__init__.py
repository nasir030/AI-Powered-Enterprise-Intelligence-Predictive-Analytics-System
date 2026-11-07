"""Enterprise intelligence package."""

from .analytics import compute_kpis, generate_business_insights
from .data_pipeline import clean_business_data, generate_demo_dataframe, infer_mapping
from .ml import (
    AnomalyDetector,
    CustomerSegmentationModel,
    ProductRecommendationEngine,
    SalesForecastModel,
)

__all__ = [
    "clean_business_data",
    "generate_demo_dataframe",
    "infer_mapping",
    "compute_kpis",
    "generate_business_insights",
    "SalesForecastModel",
    "CustomerSegmentationModel",
    "AnomalyDetector",
    "ProductRecommendationEngine",
]
