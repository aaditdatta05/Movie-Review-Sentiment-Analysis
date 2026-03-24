"""Text preprocessing utilities for sentiment analysis."""

from __future__ import annotations

import re
from typing import Iterable

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def _ensure_nltk_resources() -> None:
    """Ensure required NLTK resources are available."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
        # Newer NLTK versions may require punkt_tab for tokenization.
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                # If download fails, the downstream call will raise a clear error.
                pass


_ensure_nltk_resources()
STOP_WORDS = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """Clean and normalize text for TF-IDF modeling.

    Steps:
    1. Lowercase text
    2. Remove punctuation and special characters
    3. Tokenize
    4. Remove stopwords
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in STOP_WORDS]
    return " ".join(filtered_tokens)


def preprocess_texts(texts: Iterable[str]) -> list[str]:
    """Preprocess an iterable of strings."""
    return [preprocess_text(text) for text in texts]


def preprocess_series(series: pd.Series) -> pd.Series:
    """Preprocess a pandas Series of review text."""
    return series.fillna("").astype(str).apply(preprocess_text)
