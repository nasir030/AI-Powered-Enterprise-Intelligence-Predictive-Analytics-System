# AI-Powered Enterprise Intelligence & Predictive Analytics System

This project delivers a lightweight end-to-end analytics system for retail and business datasets. It includes:

- automated data cleaning and upload handling for CSV and Excel files
- KPI dashboards and business insight generation
- sales forecasting using a time-aware machine learning approach
- RFM-based customer segmentation with k-means clustering
- anomaly detection using Isolation Forest
- product recommendation summaries
- an AI-style business analyst that answers simple dataset questions
- FastAPI backend and Streamlit frontend

## Quick start

1. Create a virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run the FastAPI backend:
   `python main.py`
4. Run the Streamlit dashboard:
   `streamlit run app.py`

## API endpoints

- `GET /health`
- `GET /demo`
- `POST /upload`
- `POST /forecast`
- `POST /segment`
- `POST /anomalies`
- `POST /assistant`

## Notes

- Profit metrics are only included when cost or profit fields are available.
- Uploaded data is preserved in memory and cleaned without silently removing important business information.
- A demo dataset is generated automatically when no CSV or Excel file is uploaded.
