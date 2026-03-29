"""Inference pipeline for BERT-based sentiment analysis."""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_PATH = "bert_model"
MAX_LENGTH = 512


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
    return max(0.0, 1.0 - abs(probability_positive - 0.5) * 5.0)


def load_bert_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load fine-tuned BERT model and tokenizer."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model directory not found: {model_path}. "
            "Run train_bert.py first to train and save the model."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer


def predict_sentiment_bert(text: str, model, tokenizer) -> dict[str, float | str]:
    """Predict sentiment using fine-tuned BERT."""
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)

    probability_positive = float(probabilities[0][1].cpu().numpy())
    label = map_probability_to_label(probability_positive)
    confidence = compute_confidence(label, probability_positive)

    return {
        "label": label,
        "confidence": confidence,
        "positive_probability": probability_positive,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict sentiment using BERT model.")
    parser.add_argument("--text", type=str, default=None, help="Raw input review text.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to fine-tuned BERT model directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    user_text = args.text if args.text else input("Enter a movie review: ").strip()

    if not user_text:
        raise ValueError("Input text cannot be empty.")

    loaded_model, loaded_tokenizer = load_bert_model(args.model_path)
    result = predict_sentiment_bert(user_text, loaded_model, loaded_tokenizer)

    print(f"Sentiment: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Positive Probability: {result['positive_probability']:.2%}")
