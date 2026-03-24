"""Train a movie review sentiment analysis model using TF-IDF + Logistic Regression."""

from __future__ import annotations

import argparse
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from preprocess import preprocess_series


DEFAULT_DATASET_PATH = "IMDB Dataset.csv"
DEFAULT_MODEL_PATH = "model.pkl"
DEFAULT_VECTORIZER_PATH = "vectorizer.pkl"
REVIEW_COLUMN = "review"
SENTIMENT_COLUMN = "sentiment"


def load_and_validate_data(csv_path: str) -> tuple[pd.Series, pd.Series]:
    """Load dataset and validate expected columns and label values."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {REVIEW_COLUMN, SENTIMENT_COLUMN}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[[REVIEW_COLUMN, SENTIMENT_COLUMN]].dropna()
    df[REVIEW_COLUMN] = df[REVIEW_COLUMN].astype(str).str.strip()
    df[SENTIMENT_COLUMN] = df[SENTIMENT_COLUMN].astype(str).str.strip().str.lower()
    df = df[df[REVIEW_COLUMN] != ""]

    label_map = {"negative": 0, "positive": 1}
    unknown_labels = sorted(set(df[SENTIMENT_COLUMN]) - set(label_map))
    if unknown_labels:
        raise ValueError(
            "Unexpected sentiment labels found. "
            f"Expected only {sorted(label_map)} but got {unknown_labels}."
        )

    X = df[REVIEW_COLUMN]
    y = df[SENTIMENT_COLUMN].map(label_map)
    return X, y


def save_pickle(obj: object, output_path: str) -> None:
    """Save a Python object as a pickle file."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "wb") as file:
        pickle.dump(obj, file)


def train_model(dataset_path: str, model_path: str, vectorizer_path: str) -> None:
    """Run full training, evaluation, and persistence pipeline."""
    X, y = load_and_validate_data(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train_clean = preprocess_series(X_train)
    X_test_clean = preprocess_series(X_test)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train_clean)
    X_test_tfidf = vectorizer.transform(X_test_clean)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Model Evaluation")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    save_pickle(model, model_path)
    save_pickle(vectorizer, vectorizer_path)
    print(f"Saved model to: {model_path}")
    print(f"Saved vectorizer to: {vectorizer_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sentiment analysis model.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Path to dataset CSV file.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Output path for model pickle.")
    parser.add_argument(
        "--vectorizer-path",
        default=DEFAULT_VECTORIZER_PATH,
        help="Output path for vectorizer pickle.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args.dataset, args.model_path, args.vectorizer_path)
