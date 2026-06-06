from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.features import extract_numeric_features, highlighted_suspicious_terms, risk_level
from src.model import (
    CLASSIFIERS,
    MODEL_PATH,
    article_feature_influences,
    available_model_artifacts,
    dataset_quality_report,
    load_dataset,
    load_model,
    predict,
    predict_batch,
    read_csv_with_fallback,
    top_text_features,
    train_all_models,
)


st.set_page_config(page_title="Fake News Detection System", page_icon="FN", layout="wide")

st.title("Fake News Detection System")
st.caption("Machine learning and NLP support for classifying news text as Real or Fake.")


@st.cache_resource
def cached_model(path: str, model_mtime: float):
    return load_model(path)


def model_available() -> bool:
    return MODEL_PATH.exists()


def plot_confusion_matrix(matrix: list[list[int]], labels: list[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 3.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            ax.text(column_index, row_index, value, ha="center", va="center", color="#111111")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, clear_figure=True)


def plot_roc_curve(curve: dict[str, list[float]] | None, auc: float | None, title: str) -> None:
    if not curve or auc is None:
        st.info("ROC curve is unavailable for this dataset split.")
        return

    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.plot(curve["fpr"], curve["tpr"], label=f"AUC = {auc:.2f}", color="#3478a8")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig, clear_figure=True)


def format_percent(value: float | None) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2%}"


def format_number(value: float | None) -> str:
    return "N/A" if pd.isna(value) else f"{value:.3f}"


with st.sidebar:
    st.header("Training")
    st.write("Trains Logistic Regression, Naive Bayes, and Support Vector Machine together.")
    uploaded = st.file_uploader("Training CSV", type=["csv"])
    use_sample = st.checkbox("Use sample_news.csv", value=not uploaded)
    tune_models = st.checkbox("Tune hyperparameters", value=True)
    test_size = st.slider("Test set size", min_value=0.15, max_value=0.4, value=0.25, step=0.05)
    train_clicked = st.button("Train all classifiers", type="primary")

    if train_clicked:
        if uploaded:
            with NamedTemporaryFile(delete=False, suffix=".csv") as temp:
                temp.write(uploaded.getbuffer())
                data_path = temp.name
        elif use_sample and Path("sample_news.csv").exists():
            data_path = "sample_news.csv"
        else:
            st.error("Upload a CSV or enable the sample dataset.")
            st.stop()

        try:
            preview = load_dataset(data_path)
        except ValueError as error:
            st.error(str(error))
            st.stop()

        st.subheader("Dataset Preview")
        st.dataframe(preview.head(10), use_container_width=True, hide_index=True)
        st.write(f"Rows: {len(preview)}")
        st.bar_chart(preview["label"].value_counts())
        warnings = dataset_quality_report(preview)
        if warnings:
            st.warning("\n".join(warnings))
        else:
            st.success("Dataset quality checks passed.")

        with st.spinner("Training classifiers..."):
            results, best_model = train_all_models(data_path, tune=tune_models, test_size=test_size)

        comparison = pd.DataFrame(
            [
                {
                    "Classifier": result.display_name,
                    "Accuracy": result.accuracy,
                    "Precision": result.precision,
                    "Recall": result.recall,
                    "F1": result.f1,
                    "ROC-AUC": result.roc_auc,
                    "CV Accuracy": result.cv_accuracy_mean,
                    "CV Precision": result.cv_precision_mean,
                    "CV Recall": result.cv_recall_mean,
                    "CV F1": result.cv_f1_mean,
                    "Saved As Best": result.model_name == best_model,
                }
                for result in results
            ]
        ).sort_values(["F1", "Accuracy"], ascending=False)

        st.success(f"Training complete. Best model: {CLASSIFIERS[best_model]}")
        display_comparison = comparison.copy()
        for column in ["Accuracy", "Precision", "Recall", "F1", "CV Accuracy", "CV Precision", "CV Recall", "CV F1"]:
            display_comparison[column] = display_comparison[column].apply(format_percent)
        display_comparison["ROC-AUC"] = display_comparison["ROC-AUC"].apply(format_number)
        st.dataframe(
            display_comparison,
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download model comparison",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv",
        )

        chart_frame = comparison.set_index("Classifier")[["Accuracy", "Precision", "Recall", "F1"]]
        st.bar_chart(chart_frame)

        for result in sorted(results, key=lambda item: item.accuracy, reverse=True):
            with st.expander(f"{result.display_name} report"):
                if result.best_params:
                    st.write("Best parameters:", result.best_params)
                if result.cv_accuracy_mean is not None:
                    cv_frame = pd.DataFrame(
                        [
                            {
                                "Accuracy": result.cv_accuracy_mean,
                                "Precision": result.cv_precision_mean,
                                "Recall": result.cv_recall_mean,
                                "F1": result.cv_f1_mean,
                            }
                        ]
                    )
                    st.write(f"Cross-validation accuracy: {result.cv_accuracy_mean:.2%} +/- {result.cv_accuracy_std:.2%}")
                    display_cv = cv_frame.copy()
                    for column in ["Accuracy", "Precision", "Recall", "F1"]:
                        display_cv[column] = display_cv[column].apply(format_percent)
                    st.dataframe(
                        display_cv,
                        use_container_width=True,
                        hide_index=True,
                    )
                st.text(result.report)
                matrix_col, roc_col = st.columns(2)
                with matrix_col:
                    plot_confusion_matrix(result.confusion_matrix, result.labels, result.display_name)
                with roc_col:
                    plot_roc_curve(result.roc_curve, result.roc_auc, f"{result.display_name} ROC")


left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.subheader("Analyze Article")
    article_text = st.text_area(
        "News text",
        height=260,
        placeholder="Paste a news headline, article, or social media post here.",
    )
    analyze_clicked = st.button("Analyze", disabled=not article_text.strip())

with right:
    st.subheader("Model Status")
    artifacts = available_model_artifacts()
    model_options = {"best": MODEL_PATH} if model_available() else {}
    model_options.update(artifacts)
    if model_options:
        selected_model = st.selectbox(
            "Prediction model",
            list(model_options),
            format_func=lambda value: "Best saved model"
            if value == "best"
            else "Ensemble voting"
            if value == "ensemble"
            else CLASSIFIERS[value],
        )
        selected_model_path = model_options[selected_model]
        artifact = cached_model(str(selected_model_path), selected_model_path.stat().st_mtime)
        model_name = artifact.get("model_name")
        model_label = "Ensemble voting" if model_name == "ensemble" else CLASSIFIERS.get(model_name, "trained classifier")
        st.success(f"Ready: {model_label}")
    else:
        selected_model_path = MODEL_PATH
        st.warning("No trained model found. Train one from the sidebar first.")
    st.write("Expected CSV columns: `text`, `label`")
    st.write("Labels can be `real` and `fake`.")

    st.subheader("Batch Prediction")
    batch_file = st.file_uploader("Prediction CSV", type=["csv"], key="batch")
    batch_clicked = st.button("Run batch prediction", disabled=not batch_file or not model_options)


if analyze_clicked:
    if not model_available():
        st.error("Train a model before analyzing text.")
        st.stop()

    artifact = cached_model(str(selected_model_path), selected_model_path.stat().st_mtime)
    result = predict(article_text, artifact)
    fake_probability = result["fake_probability"]
    risk = risk_level(fake_probability)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Classification", result["label"].title())
    metric_cols[1].metric("Fake Probability", f"{fake_probability:.1%}")
    metric_cols[2].metric("Risk Level", risk)

    st.progress(min(1.0, max(0.0, fake_probability)))

    st.subheader("Highlighted Signals")
    st.markdown(highlighted_suspicious_terms(article_text))

    feature_frame = extract_numeric_features([article_text]).T.reset_index()
    feature_frame.columns = ["Feature", "Value"]

    detail_left, detail_right = st.columns(2, gap="large")
    with detail_left:
        st.subheader("Content Signals")
        st.dataframe(feature_frame, use_container_width=True, hide_index=True)

    with detail_right:
        st.subheader("Influential Model Features")
        important = top_text_features(artifact)
        if important.empty:
            st.info("Feature coefficients are unavailable for this classifier.")
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            plot_data = important.sort_values("absolute_importance")
            colors = ["#d04f45" if value > 0 else "#3478a8" for value in plot_data["importance"]]
            ax.barh(plot_data["feature"], plot_data["importance"], color=colors)
            ax.axvline(0, color="#333333", linewidth=0.8)
            ax.set_xlabel("Coefficient")
            ax.set_ylabel("")
            st.pyplot(fig, clear_figure=True)

    probabilities = pd.DataFrame(
        [{"Label": label.title(), "Probability": probability} for label, probability in result["probabilities"].items()]
    )
    st.subheader("Class Probabilities")
    st.bar_chart(probabilities, x="Label", y="Probability")

    st.subheader("Article-Level Explanation")
    influences = article_feature_influences(article_text, artifact)
    if influences.empty:
        st.info("Article-level coefficient explanations are unavailable for this model.")
    else:
        explanation_frame = influences.copy()
        explanation_frame["Direction"] = explanation_frame["influence"].apply(
            lambda value: "Supports fake" if value > 0 else "Supports real"
        )
        st.dataframe(
            explanation_frame[["feature", "Direction", "influence"]].rename(
                columns={"feature": "Feature", "influence": "Influence"}
            ),
            use_container_width=True,
            hide_index=True,
        )


if batch_clicked:
    if not model_options:
        st.error("Train a model before batch prediction.")
        st.stop()

    try:
        batch_data = read_csv_with_fallback(batch_file)
    except Exception as error:
        st.error(f"Could not read CSV: {error}")
        st.stop()

    if "text" not in batch_data.columns:
        st.error("Prediction CSV must contain a `text` column.")
        st.stop()

    artifact = cached_model(str(selected_model_path), selected_model_path.stat().st_mtime)
    predictions = predict_batch(batch_data["text"].fillna(""), artifact)
    st.subheader("Batch Results")
    st.dataframe(predictions, use_container_width=True, hide_index=True)
    st.bar_chart(predictions["risk_level"].value_counts())
    st.download_button(
        "Download prediction report",
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name="fake_news_predictions.csv",
        mime="text/csv",
    )
