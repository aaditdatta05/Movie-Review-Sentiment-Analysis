"""Train a BERT-based sentiment analysis model on IMDb reviews."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_DATASET_PATH = "IMDB Dataset.csv"
DEFAULT_MODEL_PATH = "bert_model"
REVIEW_COLUMN = "review"
SENTIMENT_COLUMN = "sentiment"
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 2e-5


class SentimentDataset(Dataset):
    """Torch dataset for sentiment analysis."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_and_validate_data(csv_path: str) -> tuple[list[str], list[int]]:
    """Load dataset and validate required columns and labels."""
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

    texts = df[REVIEW_COLUMN].tolist()
    labels = df[SENTIMENT_COLUMN].map(label_map).tolist()
    return texts, labels


def train_bert_model(dataset_path: str, model_output_path: str) -> None:
    """Fine-tune BERT using custom training loop and evaluate on test set."""
    print("Loading and validating dataset...")
    texts, labels = load_and_validate_data(dataset_path)

    print("Splitting data into train/test (80/20)...")
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    print("Creating training and evaluation datasets...")
    train_dataset = SentimentDataset(texts_train, labels_train, tokenizer)
    eval_dataset = SentimentDataset(texts_test, labels_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Fine-tuning BERT...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Average Training Loss: {avg_loss:.4f}")

    print("\nEvaluating on test set...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("\nBERT Model Evaluation")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    os.makedirs(model_output_path, exist_ok=True)
    model.save_pretrained(model_output_path)
    tokenizer.save_pretrained(model_output_path)
    print(f"\nModel and tokenizer saved to: {model_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT sentiment analysis model.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Path to dataset CSV file.")
    parser.add_argument("--output", default=DEFAULT_MODEL_PATH, help="Output directory for model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_bert_model(args.dataset, args.output)
