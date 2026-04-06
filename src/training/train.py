import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def train_model(X_train, X_test, y_train, y_test):
    """
    Trains an XGBoost model and logs with MLflow.

    Args:
        X_train: Training features
        X_test:  Test features
        y_train: Training labels
        y_test:  Test labels
    """
    model = XGBClassifier(
        n_estimators=311,
        learning_rate=0.01,
        max_depth=4,
        random_state=42,
        scale_pose_weight = 2.5,
        n_jobs=-1,
        eval_metric="logloss"
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        # Log params, metrics and model
        mlflow.log_param("n_estimators", 311)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.xgboost.log_model(model, name="model",
                                 registered_model_name="churn-model")

        # Log dataset so it shows in MLflow UI
        train_ds = mlflow.data.from_pandas(
            pd.concat([X_train, y_train], axis=1),
            source="training_data"
        )
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")


if __name__ == "__main__":
    X_train = pd.read_csv("data/processed/X_train.csv").astype(float)
    X_test  = pd.read_csv("data/processed/X_test.csv").astype(float)
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

    train_model(X_train, X_test, y_train, y_test)