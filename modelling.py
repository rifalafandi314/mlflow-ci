import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# LOAD DATA (TEXT!)
# =========================
X_train = joblib.load("dataset_preprocessing/X_train.pkl")
X_test = joblib.load("dataset_preprocessing/X_test.pkl")
y_train = joblib.load("dataset_preprocessing/y_train.pkl")
y_test = joblib.load("dataset_preprocessing/y_test.pkl")

# =========================
# SET EXPERIMENT
# =========================
mlflow.set_experiment("sentiment_tuning_rf")

# =========================
# PARAM GRID
# =========================
param_grid = {
    "clf__n_estimators": [50, 100],
    "clf__max_depth": [None, 10]
}

# =========================
# LOOP
# =========================
for params in ParameterGrid(param_grid):

    with mlflow.start_run():

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
            ("clf", RandomForestClassifier(random_state=42))
        ])

        pipeline.set_params(**params)

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        # metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # logging
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(pipeline, "model")

        print("Params:", params)
        print("Accuracy:", acc)