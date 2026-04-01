import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# CLEAN ENV (WAJIB)
# =========================
os.environ.pop("MLFLOW_RUN_ID", None)
os.environ.pop("MLFLOW_EXPERIMENT_ID", None)

# =========================
# TRACKING (CI SAFE)
# =========================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("sentiment-experiment")

# =========================
# LOAD DATA
# =========================
X_train = joblib.load("dataset_preprocessing/X_train.pkl")
X_test = joblib.load("dataset_preprocessing/X_test.pkl")
y_train = joblib.load("dataset_preprocessing/y_train.pkl")
y_test = joblib.load("dataset_preprocessing/y_test.pkl")

# =========================
# TRAINING
# =========================
with mlflow.start_run(run_name="random_forest_training"):

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # logging
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    # log model (AMAN)
    mlflow.sklearn.log_model(
        sk_model=model,
        name="random_forest_model"
    )

    print("Accuracy:", acc)