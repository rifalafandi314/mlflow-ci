import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 🔥 FIX WAJIB
os.environ.pop("MLFLOW_RUN_ID", None)
mlflow.set_tracking_uri("file:./mlruns")

X_train = joblib.load("dataset_preprocessing/X_train.pkl")
X_test = joblib.load("dataset_preprocessing/X_test.pkl")
y_train = joblib.load("dataset_preprocessing/y_train.pkl")
y_test = joblib.load("dataset_preprocessing/y_test.pkl")

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)