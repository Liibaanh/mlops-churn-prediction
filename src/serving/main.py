from fastapi import FastAPI
from pydantic import BaseModel
# import Gradio as gr
# from .... # ML model

# Initialize FastAPI application
app = FastAPI(
    title="Telco Custiomer Churn API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0"
)

# CRITICAL: Required for AWS Application Load Balancer health checks
@app.get("/")
def root():
    """
    check endpoint for monitoring and load balancer health checks.
    """
    return {"status":"ok"}


# Pydantic model for automatic and API documentation
# Validate, document and serv
class CustomerData(BaseModel):
    
    """
    Customer data schema for churn prediction.
    
    This schema defines the exact 18 features required for churn prediction.
    All features match the original dataset structure for consistency.
    
    """
    # Demographics 
    gender : str                # "Male" or "Female"
    Partner : str               # "Yes" or "No"
    Dependents : str            # "Yes" or "No"
    
    # Phone services 
    PhoneService : str          # "Yes" or "No" 
    MultipleLines : str         # "Yes", "No" "No Phone service"
    
    # Internet services
    InternetServices : str      # "DSL", "Fiber optic", or "No"
    OnlineSecurity : str        # "Yes", "No" or "No internet service"
    OnlineBackup : str          # "Yes", "No" or "No internet service"
    DeviceProtection : str      # "Yes", "No" or "No internet service"
    TechSupport : str           # "Yes", "No" or "No internet service"
    StreamingTV : str           # "Yes", "No" or "No internet service"
    StreamingMovies : str       # "Yes", "No" or "No internet service"
    
    # Account information
    Contract : str              # "Month-to-Month", "One year", "Two year"
    PaperlessBilling : str      # "Yes" or "No"
    PaymentMethod : str         # "Electronic check", "Mailed check", etc
    
    # Numeric features
    tenure : int                # Number of months with company
    MonthlyCharges : float      # Monthly charges in dollars
    TotalCharges : float        # Total charges to date
    
    
@app.post("/predict")

def get_prediction(data: CustomerData):
    """
    Main prediction endpoint for customer churn prediction
    
    This endpoint:
    1. Receive validated customer data via pydantic model
    2. Calls the ... pipeline to transform features and predict
    3. Returns churn prediction in JSON format
    
    Expected Response
    - {"Prediction": "Likely to churn"} or {"prediction": "Not likely to churn"}
    - {"error":"error_message"} if prediction fails

    """
    
    try:
        # Convert Pydantic model to dict and call ... pipeline
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        # Return error details for debugging
        return {"error" : str(e)}
    
    