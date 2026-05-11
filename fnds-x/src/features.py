import re
from typing import Iterable

import numpy as np
import pandas as pd


SUSPICIOUS_KEYWORDS = {
    "shocking",
    "secret",
    "miracle",
    "hidden",
    "exposed",
    "conspiracy",
    "hoax",
    "cure",
    "unnamed",
    "viral",
}

POSITIVE_WORDS = {
    "approved",
    "confirmed",
    "safe",
    "benefit",
    "success",
    "reliable",
    "verified",
    "improve",
}

NEGATIVE_WORDS = {
    "fake",
    "threat",
    "danger",
    "secret",
    "shocking",
    "fear",
    "crisis",
    "hidden",
    "corrupt",
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s.!?]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _count_words(text: str, words: Iterable[str]) -> int:
    tokens = set(clean_text(text).split())
    return sum(1 for word in words if word in tokens)


def extract_numeric_features(texts: Iterable[str]) -> pd.DataFrame:
    rows = []
    for text in texts:
        raw = str(text)
        cleaned = clean_text(raw)
        words = cleaned.split()
        sentence_count = max(1, len(re.findall(r"[.!?]+", raw)))
        uppercase_words = re.findall(r"\b[A-Z]{2,}\b", raw)
        positive = _count_words(raw, POSITIVE_WORDS)
        negative = _count_words(raw, NEGATIVE_WORDS)
        suspicious = _count_words(raw, SUSPICIOUS_KEYWORDS)
        rows.append(
            {
                "char_count": len(raw),
                "word_count": len(words),
                "sentence_count": sentence_count,
                "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
                "avg_sentence_length": len(words) / sentence_count,
                "exclamation_count": raw.count("!"),
                "question_count": raw.count("?"),
                "uppercase_word_count": len(uppercase_words),
                "suspicious_keyword_count": suspicious,
                "sentiment_score": (positive - negative) / max(1, positive + negative),
            }
        )
    return pd.DataFrame(rows)


def risk_level(probability_fake: float) -> str:
    if probability_fake >= 0.75:
        return "High"
    if probability_fake >= 0.45:
        return "Medium"
    return "Low"


def highlighted_suspicious_terms(text: str) -> str:
    highlighted = str(text)
    for keyword in sorted(SUSPICIOUS_KEYWORDS, key=len, reverse=True):
        highlighted = re.sub(
            rf"\b({re.escape(keyword)})\b",
            r"**\1**",
            highlighted,
            flags=re.IGNORECASE,
        )
    return highlighted
