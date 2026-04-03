import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load data
df = pd.read_csv("dataset_preprocessing/data_clean.csv")

df = df.dropna(subset=['clean_text'])
df['clean_text'] = df['clean_text'].astype(str)

X = df['clean_text']
y = df['label']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# vectorizer
tfidf = TfidfVectorizer()
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# aktifkan autolog
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)