import pandas as pd
import pytest

from src.model import (
    available_model_artifacts,
    dataset_quality_report,
    load_dataset,
    load_model,
    predict,
    read_csv_with_fallback,
    train_all_models,
)


def test_load_dataset_rejects_invalid_labels(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["one", "two", "three", "four"], "label": ["real", "fake", "maybe", "real"]}).to_csv(
        path, index=False
    )

    with pytest.raises(ValueError):
        load_dataset(path)


def test_load_dataset_accepts_zero_one_labels(tmp_path):
    path = tmp_path / "numeric.csv"
    pd.DataFrame({"text": ["one", "two", "three", "four"], "label": [0, 1, 0, 1]}).to_csv(path, index=False)

    data = load_dataset(path)

    assert data["label"].tolist() == ["real", "fake", "real", "fake"]


def test_load_dataset_accepts_float_zero_one_labels(tmp_path):
    path = tmp_path / "numeric_float.csv"
    pd.DataFrame({"text": ["one", "two", "three", "four"], "label": [0.0, 1.0, 0.0, 1.0]}).to_csv(
        path, index=False
    )

    data = load_dataset(path)

    assert data["label"].tolist() == ["real", "fake", "real", "fake"]


def test_read_csv_with_fallback_accepts_cp1252(tmp_path):
    path = tmp_path / "encoded.csv"
    path.write_bytes("text,label\nSmart quote \u201cnews\u201d,1\nPlain report,0\n".encode("cp1252"))

    data = read_csv_with_fallback(path)

    assert list(data.columns) == ["text", "label"]
    assert len(data) == 2


def test_train_all_models_creates_predictable_artifact(tmp_path, monkeypatch):
    path = tmp_path / "news.csv"
    pd.DataFrame(
        {
            "text": [
                "Verified report from officials",
                "Government published the data",
                "Scientists confirmed the results",
                "Local agency approved the update",
                "Secret miracle cure exposed",
                "Shocking hidden conspiracy claim",
                "Unnamed doctors reveal viral hoax",
                "Fake story claims magic software",
            ],
            "label": ["real", "real", "real", "real", "fake", "fake", "fake", "fake"],
        }
    ).to_csv(path, index=False)

    monkeypatch.chdir(tmp_path)
    results, best_model = train_all_models(path, tune=False)

    assert len(results) == 4
    assert best_model in {"logistic_regression", "naive_bayes", "svm", "perceptron"}
    assert "ensemble" in available_model_artifacts()
    assert "perceptron" in available_model_artifacts()
    prediction = predict("Secret miracle cure from unnamed doctors", load_model())
    assert prediction["label"] in {"real", "fake"}
    assert 0.0 <= prediction["fake_probability"] <= 1.0


def test_dataset_quality_report_flags_duplicates_and_short_text():
    data = pd.DataFrame(
        {
            "text": ["short", "short", "This verified report contains enough words"],
            "label": ["fake", "fake", "real"],
        }
    )
    warnings = dataset_quality_report(data)
    assert any("duplicate" in warning for warning in warnings)
    assert any("very short" in warning for warning in warnings)
