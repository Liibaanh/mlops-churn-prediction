import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score


def train_model(df: pd.DataFrame, target_col: str):
    '''
    Trains an XGBoost model and logs with MLflow
    
    Args:
        df (pd.DataFrame) : Feature dataset
        target_col (str) : Name of the target column.
    '''
    
    x = df.drop(columns=[target_col])
    y = df[target_col]
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = XGBClassifier(
        n_estimators = 311,
        learning_rate = 0.01,
        max_depth =4,
        random_state=42,
        n_jobs =-1,
        eval_metric="logloss"
    )
    
    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        
        # Log params, metrics and model
        mlflow.log_param("n_estimators", 311)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.xgboost.log_model(model, "model")
        
        # Log dataset so it shows in MLflow UI
        train_ds = mlflow.data.from_pandas(df, sources="training_data")
        mlflow.log_input(train_ds, context="training")
        
        print(f"Model trained. Accuracy: {acc:.4f}, Recall {rec:.4f}")
        
        
