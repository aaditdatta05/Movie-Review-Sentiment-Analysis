"""Inference pipeline for movie review sentiment analysis."""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Any

from preprocess import preprocess_text


DEFAULT_MODEL_PATH = "model.pkl"
DEFAULT_VECTORIZER_PATH = "vectorizer.pkl"


def load_pickle(file_path: str) -> Any:
    """Load and return a pickled object."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")
    with open(file_path, "rb") as file:
        return pickle.load(file)


def load_artifacts(model_path: str = DEFAULT_MODEL_PATH, vectorizer_path: str = DEFAULT_VECTORIZER_PATH):
    """Load model and vectorizer artifacts."""
    model = load_pickle(model_path)
    vectorizer = load_pickle(vectorizer_path)
    return model, vectorizer


def map_probability_to_label(probability_positive: float) -> str:
    """Map positive-class probability to Positive/Negative/Neutral."""
    if probability_positive > 0.6:
        return "Positive"
    if probability_positive < 0.4:
        return "Negative"
    return "Neutral"


def compute_confidence(label: str, probability_positive: float) -> float:
    """Compute label confidence for display."""
    if label == "Positive":
        return probability_positive
    if label == "Negative":
        return 1.0 - probability_positive
    # For neutral, confidence is highest near 0.5 and lower near threshold edges.
    return max(0.0, 1.0 - abs(probability_positive - 0.5) * 5.0)


def predict_sentiment(text: str, model: Any, vectorizer: Any) -> dict[str, float | str]:
    """Predict sentiment and return label with scores."""
    cleaned_text = preprocess_text(text)
    features = vectorizer.transform([cleaned_text])
    probability_positive = float(model.predict_proba(features)[0][1])

    label = map_probability_to_label(probability_positive)
    confidence = compute_confidence(label, probability_positive)

    return {
        "label": label,
        "confidence": confidence,
        "positive_probability": probability_positive,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict sentiment for a movie review.")
    parser.add_argument("--text", type=str, default=None, help="Raw input review text.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to model pickle.")
    parser.add_argument(
        "--vectorizer-path",
        default=DEFAULT_VECTORIZER_PATH,
        help="Path to vectorizer pickle.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    user_text = args.text if args.text else input("Enter a movie review: ").strip()

    if not user_text:
        raise ValueError("Input text cannot be empty.")

    loaded_model, loaded_vectorizer = load_artifacts(args.model_path, args.vectorizer_path)
    result = predict_sentiment(user_text, loaded_model, loaded_vectorizer)

    print(f"Sentiment: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Positive Probability: {result['positive_probability']:.2%}")
