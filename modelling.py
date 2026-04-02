import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid

# =========================
# LOAD DATA (TF-IDF MATRIX)
# =========================
X_train = joblib.load("dataset_preprocessing/X_train.pkl")
X_test = joblib.load("dataset_preprocessing/X_test.pkl")
y_train = joblib.load("dataset_preprocessing/y_train.pkl")
y_test = joblib.load("dataset_preprocessing/y_test.pkl")

# =========================
# PARAM GRID
# =========================
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10]
}

# =========================
# LOOP
# =========================
for params in ParameterGrid(param_grid):

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=42
    )

    # TRAIN
    model.fit(X_train, y_train)

    # PREDICT
    y_pred = model.predict(X_test)

    # METRICS
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # LOG
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

    print("Params:", params, "Accuracy:", acc)