from fastapi.testclient import TestClient
from src.serving.main import app
import pytest

client = TestClient(app)

def test_prediction():
    # This is test_data 
    test_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

    response = client.post("/predict", json=test_data)
    
    # Sjekker at serveren svarer OK
    assert response.status_code == 200
    
    # Sjekker at svaret inneholder en prediksjon (f.eks. 0 eller 1)
    json_response = response.json()
    assert "prediction" in json_response