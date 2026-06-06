from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, MinMaxScaler
from sklearn.svm import LinearSVC

from .features import extract_numeric_features


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "fake_news_model.joblib"
ENSEMBLE_MODEL_PATH = MODEL_DIR / "fake_news_ensemble.joblib"
CLASSIFIERS = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "svm": "Support Vector Machine",
    "perceptron": "Perceptron",
}
LABEL_MAP = {
    "0": "real",
    "0.0": "real",
    "1": "fake",
    "1.0": "fake",
    "real": "real",
    "fake": "fake",
    "true": "real",
    "false": "fake",
}
CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin1")


def read_csv_with_fallback(csv_path: str | Path) -> pd.DataFrame:
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(f"Could not read CSV with supported encodings: {', '.join(CSV_ENCODINGS)}") from last_error


def _text_column(frame: pd.DataFrame) -> pd.Series:
    return frame["text"].fillna("")


def _numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    return extract_numeric_features(frame["text"].fillna(""))


def build_pipeline(model_name: str = "logistic_regression") -> Pipeline:
    text_features = TfidfVectorizer(
        preprocessor=None,
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000,
    )

    features = ColumnTransformer(
        transformers=[
            ("tfidf", text_features, "text"),
            (
                "numeric",
                Pipeline(
                    [
                        ("extract", FunctionTransformer(_numeric_features, validate=False)),
                        ("scale", MinMaxScaler()),
                    ]
                ),
                ["text"],
            ),
        ]
    )

    if model_name == "naive_bayes":
        classifier: Any = MultinomialNB()
    elif model_name == "svm":
        classifier = LinearSVC()
    elif model_name == "perceptron":
        classifier = Perceptron(max_iter=1000, random_state=42)
    else:
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced")

    return Pipeline([("features", features), ("classifier", classifier)])


@dataclass
class TrainResult:
    model_name: str
    display_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    cv_accuracy_mean: float | None
    cv_accuracy_std: float | None
    cv_precision_mean: float | None
    cv_recall_mean: float | None
    cv_f1_mean: float | None
    labels: list[str]
    report: str
    confusion_matrix: list[list[int]]
    roc_curve: dict[str, list[float]] | None
    best_params: dict[str, Any]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    data = read_csv_with_fallback(csv_path)
    required = {"text", "label"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
    data = data.dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(str).str.lower().str.strip().map(LABEL_MAP)
    if data["label"].isna().any():
        bad_values = read_csv_with_fallback(csv_path)["label"].astype(str).str.lower().str.strip()
        invalid_labels = sorted(str(value) for value in set(bad_values[~bad_values.isin(LABEL_MAP)]))
        raise ValueError(
            "Labels must be real/fake or 0/1. "
            "This project maps 0=real and 1=fake. "
            f"Found: {', '.join(invalid_labels)}"
        )
    invalid_labels = sorted(set(data["label"]).difference({"real", "fake"}))
    if invalid_labels:
        raise ValueError(f"Labels must be real or fake. Found: {', '.join(invalid_labels)}")
    if data["label"].nunique() < 2:
        raise ValueError("CSV must contain both real and fake examples.")
    if len(data) < 4:
        raise ValueError("CSV must contain at least 4 rows.")
    return data


def dataset_quality_report(data: pd.DataFrame) -> list[str]:
    warnings = []
    duplicate_count = int(data["text"].duplicated().sum())
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate text row(s) found.")

    short_count = int(data["text"].str.split().str.len().lt(5).sum())
    if short_count:
        warnings.append(f"{short_count} very short article(s) have fewer than 5 words.")

    label_counts = data["label"].value_counts()
    minority_ratio = float(label_counts.min() / label_counts.sum())
    if minority_ratio < 0.3:
        warnings.append("Dataset is imbalanced; one class is below 30% of the rows.")

    empty_after_strip = int(data["text"].str.strip().eq("").sum())
    if empty_after_strip:
        warnings.append(f"{empty_after_strip} empty text row(s) found after trimming.")
    return warnings


def _split_dataset(
    data: pd.DataFrame, test_size: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(data["label"])

    stratify = encoded_labels if len(np.unique(encoded_labels)) > 1 and len(data) >= 6 else None
    if stratify is not None:
        class_count = len(np.unique(encoded_labels))
        test_size = max(test_size, class_count / len(data))
        test_size = min(test_size, 1 - (class_count / len(data)))
    x_train, x_test, y_train, y_test = train_test_split(
        data[["text"]],
        encoded_labels,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )
    return x_train, x_test, y_train, y_test, encoder


def _model_probabilities(pipeline: Pipeline, frame: pd.DataFrame, class_count: int) -> np.ndarray:
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        return pipeline.predict_proba(frame)
    if hasattr(classifier, "decision_function"):
        scores = np.asarray(pipeline.decision_function(frame))
        if scores.ndim == 1:
            positive = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - positive, positive])
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    predictions = pipeline.predict(frame)
    probabilities = np.zeros((len(frame), class_count))
    probabilities[np.arange(len(frame)), predictions] = 1.0
    return probabilities


def _positive_class_index(encoder: LabelEncoder) -> int | None:
    classes = list(encoder.classes_)
    return classes.index("fake") if "fake" in classes else None


def _train_on_split(
    model_name: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    encoder: LabelEncoder,
    save_path: str | Path | None = None,
    tune: bool = True,
) -> tuple[TrainResult, dict[str, Any]]:
    if model_name not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier: {model_name}")

    pipeline = build_pipeline(model_name)
    best_params = {}
    class_counts = np.bincount(y_train)
    min_class_count = int(class_counts.min()) if len(class_counts) else 0

    if tune and min_class_count >= 2:
        param_grid = {
            "logistic_regression": {"classifier__C": [0.5, 1.0, 2.0]},
            "naive_bayes": {"classifier__alpha": [0.5, 1.0, 1.5]},
            "svm": {"classifier__C": [0.5, 1.0, 2.0]},
            "perceptron": {
                "classifier__alpha": [0.0001, 0.001, 0.01],
                "classifier__eta0": [0.1, 1.0],
            },
        }[model_name]
        cv = StratifiedKFold(n_splits=min(3, min_class_count), shuffle=True, random_state=42)
        search = GridSearchCV(pipeline, param_grid=param_grid, scoring="f1_macro", cv=cv, n_jobs=None)
        search.fit(x_train, y_train)
        pipeline = search.best_estimator_
        best_params = search.best_params_
    else:
        pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = _model_probabilities(pipeline, x_test, len(encoder.classes_))
    fake_index = _positive_class_index(encoder)

    auc = None
    curve = None
    if fake_index is not None and len(np.unique(y_test)) == 2:
        fake_scores = probabilities[:, fake_index]
        auc = float(roc_auc_score(y_test == fake_index, fake_scores))
        fpr, tpr, thresholds = roc_curve(y_test == fake_index, fake_scores)
        curve = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}

    cv_mean = cv_std = cv_precision = cv_recall = cv_f1 = None
    full_class_counts = np.bincount(np.concatenate([y_train, y_test]))
    full_min_class_count = int(full_class_counts.min()) if len(full_class_counts) else 0
    if full_min_class_count >= 2:
        cv = StratifiedKFold(n_splits=min(5, full_min_class_count), shuffle=True, random_state=42)
        cv_scores = cross_validate(
            pipeline,
            pd.concat([x_train, x_test]),
            np.concatenate([y_train, y_test]),
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "precision": make_scorer(precision_score, average="macro", zero_division=0),
                "recall": make_scorer(recall_score, average="macro", zero_division=0),
                "f1": make_scorer(f1_score, average="macro", zero_division=0),
            },
        )
        cv_mean = float(cv_scores["test_accuracy"].mean())
        cv_std = float(cv_scores["test_accuracy"].std())
        cv_precision = float(cv_scores["test_precision"].mean())
        cv_recall = float(cv_scores["test_recall"].mean())
        cv_f1 = float(cv_scores["test_f1"].mean())

    artifact = {"pipeline": pipeline, "encoder": encoder, "model_name": model_name}
    if save_path:
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(artifact, save_path)

    labels = list(encoder.classes_)
    result = TrainResult(
        model_name=model_name,
        display_name=CLASSIFIERS[model_name],
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        recall=float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        f1=float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        roc_auc=auc,
        cv_accuracy_mean=cv_mean,
        cv_accuracy_std=cv_std,
        cv_precision_mean=cv_precision,
        cv_recall_mean=cv_recall,
        cv_f1_mean=cv_f1,
        labels=labels,
        report=classification_report(y_test, predictions, target_names=labels, zero_division=0),
        confusion_matrix=confusion_matrix(y_test, predictions).tolist(),
        roc_curve=curve,
        best_params=best_params,
    )
    return result, artifact


def train_model(csv_path: str | Path, model_name: str = "logistic_regression", test_size: float = 0.25) -> TrainResult:
    data = load_dataset(csv_path)
    x_train, x_test, y_train, y_test, encoder = _split_dataset(data, test_size)
    result, _ = _train_on_split(model_name, x_train, x_test, y_train, y_test, encoder, MODEL_PATH)
    return result


def train_all_models(csv_path: str | Path, tune: bool = True, test_size: float = 0.25) -> tuple[list[TrainResult], str]:
    data = load_dataset(csv_path)
    x_train, x_test, y_train, y_test, encoder = _split_dataset(data, test_size)

    results = []
    artifacts = {}
    for model_name in CLASSIFIERS:
        model_path = MODEL_DIR / f"fake_news_{model_name}.joblib"
        result, artifact = _train_on_split(
            model_name,
            x_train,
            x_test,
            y_train,
            y_test,
            encoder,
            model_path,
            tune,
        )
        results.append(result)
        artifacts[model_name] = artifact

    ensemble_artifact = {"models": artifacts, "encoder": encoder, "model_name": "ensemble"}
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(ensemble_artifact, ENSEMBLE_MODEL_PATH)

    best_result = max(results, key=lambda item: (item.f1, item.accuracy))
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(artifacts[best_result.model_name], MODEL_PATH)
    return results, best_result.model_name


def load_model(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    return joblib.load(path)


def available_model_artifacts() -> dict[str, Path]:
    artifacts = {}
    if ENSEMBLE_MODEL_PATH.exists():
        artifacts["ensemble"] = ENSEMBLE_MODEL_PATH
    for model_name in CLASSIFIERS:
        model_path = MODEL_DIR / f"fake_news_{model_name}.joblib"
        if model_path.exists():
            artifacts[model_name] = model_path
    return artifacts


def predict(text: str, artifact: dict[str, Any]) -> dict[str, Any]:
    encoder = artifact["encoder"]
    frame = pd.DataFrame({"text": [text]})

    if "models" in artifact:
        all_probabilities = [
            _model_probabilities(model_artifact["pipeline"], frame, len(encoder.classes_))[0]
            for model_artifact in artifact["models"].values()
        ]
        probabilities = np.mean(all_probabilities, axis=0)
        predicted_index = int(np.argmax(probabilities))
    else:
        pipeline = artifact["pipeline"]
        probabilities = _model_probabilities(pipeline, frame, len(encoder.classes_))[0]
        predicted_index = int(np.argmax(probabilities))
    label = str(encoder.inverse_transform([predicted_index])[0])

    class_probabilities = {
        str(encoder.inverse_transform([index])[0]): float(probability)
        for index, probability in enumerate(probabilities)
    }
    fake_probability = class_probabilities.get("fake", float(probabilities[predicted_index]))

    return {
        "label": label,
        "probabilities": class_probabilities,
        "fake_probability": fake_probability,
    }


def predict_batch(texts: Iterable[str], artifact: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for text in texts:
        result = predict(str(text), artifact)
        rows.append(
            {
                "text": text,
                "prediction": result["label"],
                "fake_probability": result["fake_probability"],
                "real_probability": result["probabilities"].get("real", 0.0),
                "risk_level": "High" if result["fake_probability"] >= 0.75 else "Medium"
                if result["fake_probability"] >= 0.45
                else "Low",
            }
        )
    return pd.DataFrame(rows)


def top_text_features(artifact: dict[str, Any], limit: int = 15) -> pd.DataFrame:
    if "models" in artifact:
        artifact = artifact["models"].get("logistic_regression") or next(iter(artifact["models"].values()))

    pipeline = artifact["pipeline"]
    classifier = pipeline.named_steps["classifier"]
    feature_block = pipeline.named_steps["features"]
    tfidf = feature_block.named_transformers_["tfidf"]
    numeric_names = list(extract_numeric_features(["sample"]).columns)
    names = list(tfidf.get_feature_names_out()) + numeric_names

    if not hasattr(classifier, "coef_"):
        return pd.DataFrame(columns=["feature", "importance"])

    coefficients = classifier.coef_[0]
    size = min(len(names), len(coefficients))
    ranking = pd.DataFrame({"feature": names[:size], "importance": coefficients[:size]})
    ranking["absolute_importance"] = ranking["importance"].abs()
    return ranking.sort_values("absolute_importance", ascending=False).head(limit)


def article_feature_influences(text: str, artifact: dict[str, Any], limit: int = 12) -> pd.DataFrame:
    if "models" in artifact:
        artifact = artifact["models"].get("logistic_regression") or next(iter(artifact["models"].values()))

    pipeline = artifact["pipeline"]
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        return pd.DataFrame(columns=["feature", "influence"])

    feature_block = pipeline.named_steps["features"]
    transformed = feature_block.transform(pd.DataFrame({"text": [text]}))
    vector = transformed.toarray()[0] if hasattr(transformed, "toarray") else np.asarray(transformed)[0]

    tfidf = feature_block.named_transformers_["tfidf"]
    numeric_names = list(extract_numeric_features(["sample"]).columns)
    names = list(tfidf.get_feature_names_out()) + numeric_names
    coefficients = classifier.coef_[0]
    size = min(len(names), len(coefficients), len(vector))

    contributions = vector[:size] * coefficients[:size]
    ranking = pd.DataFrame({"feature": names[:size], "influence": contributions})
    ranking = ranking[ranking["influence"] != 0].copy()
    ranking["absolute_influence"] = ranking["influence"].abs()
    return ranking.sort_values("absolute_influence", ascending=False).head(limit)
