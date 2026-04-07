import os
import pandas as pd
import mlflow
import mlflow.xgboost
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

# Initialize FastAPI application
app = FastAPI(
    title="Telco Customer Churn API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0"
)

model_path = "/app/mlruns/0/models"

def load_real_model():
    if os.path.exists(model_path) and os.listdir(model_path):
        # Find the lastest version
        versions = sorted(os.listdir(model_path))
        latest_version = versions[-1]
        artifact_path = os.path.join(model_path, latest_version, "artifacts")
        print(f"Downloading the REAL model! from {artifact_path}")
        return mlflow.xgboost.load_model(artifact_path)
    return None
            
model = load_real_model()

if model is None:
    print("Model not found, creating dummy model!")
    class DummyModel:
        def __init__(self):
            self.feature_names_in_ = [ "gender", "SeniorCitizen", "Partner", "Dependents", 
                "tenure", "PhoneService", "MultipleLines", "InternetService", 
                "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                "TechSupport", "StreamingTV", "StreamingMovies", "Contract", 
                "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"    
            ]
        def predict(self, df):
            return np.zeros(len(df))
    @property
    def feature_names_in_(self):
        return [] 

    model = DummyModel()
    
    
# CRITICAL: Required for AWS Application Load Balancer health checks
@app.get("/")
def root():
    """Health check endpoint for monitoring and load balancer health checks."""
    return {"status": "ok"}


# Pydantic model — raw data from user
class CustomerData(BaseModel):
    gender: str              # "Male" or "Female"
    SeniorCitizen: int       # 0 or 1
    Partner: str             # "Yes" or "No"
    Dependents: str          # "Yes" or "No"
    tenure: int              # months with company
    PhoneService: str        # "Yes" or "No"
    MultipleLines: str       # "Yes", "No", "No phone service"
    InternetService: str     # "DSL", "Fiber optic", "No"
    OnlineSecurity: str      # "Yes", "No", "No internet service"
    OnlineBackup: str        # "Yes", "No", "No internet service"
    DeviceProtection: str    # "Yes", "No", "No internet service"
    TechSupport: str         # "Yes", "No", "No internet service"
    StreamingTV: str         # "Yes", "No", "No internet service"
    StreamingMovies: str     # "Yes", "No", "No internet service"
    Contract: str            # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str    # "Yes" or "No"
    PaymentMethod: str       # "Electronic check", "Mailed check", etc.
    MonthlyCharges: float
    TotalCharges: float


def transform(data: CustomerData) -> pd.DataFrame:
    """
    Transform raw customer data to match training features.
    Same logic as build_features.py.
    """
    df = pd.DataFrame([data.dict()])

    # Binary Yes/No → 0/1
    BINARY_COLS = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in BINARY_COLS:
        df[col] = df[col].map(lambda x: 1 if x == "Yes" else 0)

    # One-hot encode categorical columns
    CATEGORICAL_COLS = ["gender", "MultipleLines", "InternetService",
                        "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Add missing columns with 0 (in case user sends a category not seen before)
    expected_cols = model.feature_names_in_
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training data exactly
    df = df[expected_cols]

    return df


@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Main prediction endpoint for customer churn prediction.
    Accepts raw customer data, transforms it, and returns churn prediction.
    """
    try:
        df = transform(data)
        prediction = model.predict(df)
        result = "Likely to churn" if prediction[0] == 1 else "Not likely to churn"
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}