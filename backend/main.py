from __future__ import annotations

from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from enterprise_intelligence.analytics import compute_kpis, generate_business_insights
from enterprise_intelligence.business_assistant import BusinessAssistant
from enterprise_intelligence.data_pipeline import clean_business_data, generate_demo_dataframe, process_uploaded_file
from enterprise_intelligence.ml import AnomalyDetector, CustomerSegmentationModel, ProductRecommendationEngine, SalesForecastModel

app = FastAPI(title="AI Enterprise Intelligence API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/demo")
def demo_data() -> Dict[str, object]:
    data = generate_demo_dataframe()
    return {
        "summary": compute_kpis(data),
        "insights": generate_business_insights(data),
        "rows": data.head(10).to_dict(orient="records"),
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, object]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file is required.")
    try:
        file.file.seek(0)
        payload = process_uploaded_file(file.file)
        df = payload["data"]
        return {
            "report": payload["report"],
            "summary": compute_kpis(df),
            "insights": generate_business_insights(df),
            "rows": df.head(10).to_dict(orient="records"),
        }
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/forecast")
def forecast_sales(payload: Dict[str, object]) -> Dict[str, object]:
    data = payload.get("data", [])
    if not data:
        raise HTTPException(status_code=400, detail="No sales data provided.")
    df = __import__("pandas").DataFrame(data)
    series = df.set_index("date")["sales"] if "date" in df.columns and "sales" in df.columns else df.iloc[:, 0]
    forecast = SalesForecastModel().forecast(series, horizon=int(payload.get("horizon", 30)), frequency=str(payload.get("frequency", "daily")))
    return {
        "model": forecast["model_name"],
        "metrics": forecast["metrics"],
        "forecast": forecast["forecast"].reset_index().rename(columns={"index": "date", 0: "forecast"}).to_dict(orient="records"),
    }


@app.post("/segment")
def segment_customers(payload: Dict[str, object]) -> Dict[str, object]:
    data = payload.get("data", [])
    if not data:
        raise HTTPException(status_code=400, detail="No customer data provided.")
    df = __import__("pandas").DataFrame(data)
    model = CustomerSegmentationModel()
    result = model.segment(df)
    return {"segments": result["segments"].to_dict(orient="records"), "summary": result["summary"].to_dict()}


@app.post("/anomalies")
def detect_anomalies(payload: Dict[str, object]) -> Dict[str, object]:
    data = payload.get("data", [])
    if not data:
        raise HTTPException(status_code=400, detail="No data provided.")
    df = __import__("pandas").DataFrame(data)
    result = AnomalyDetector().detect(df)
    return {"anomalies": result["anomalies"].to_dict(orient="records")}


@app.post("/assistant")
def assistant_question(payload: Dict[str, object]) -> Dict[str, object]:
    data = payload.get("data", [])
    question = payload.get("question", "")
    if not data:
        return {"answer": "No data is available for analysis."}
    df = __import__("pandas").DataFrame(data)
    return {"answer": BusinessAssistant(df).answer(str(question))}
