import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# LOAD DATA
# =========================
X_train = joblib.load("dataset_preprocessing/X_train.pkl")
X_test = joblib.load("dataset_preprocessing/X_test.pkl")
y_train = joblib.load("dataset_preprocessing/y_train.pkl")
y_test = joblib.load("dataset_preprocessing/y_test.pkl")

# =========================
# TF-IDF (WAJIB)
# =========================
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# =========================
# MLFLOW RUN
# =========================
with mlflow.start_run():

    print("Training model...")

    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # =========================
    # LOG KE MLFLOW
    # =========================
    mlflow.log_metric("accuracy", acc)

    # NOTE: pakai "name" bukan artifact_path (biar ga warning)
    mlflow.sklearn.log_model(model, name="model")

    # =========================
    # SAVE MODEL + VECTORIZER
    # =========================
    os.makedirs("model", exist_ok=True)

    joblib.dump(model, "model/model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")

    print("\nModel & vectorizer berhasil disimpan di folder 'model/'")