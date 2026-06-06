# Fake News Detection System

This project uses machine learning and natural language processing to classify news text as `real` or `fake`. It supports CSV-based training, TF-IDF and n-gram text features, engineered content signals, probability output, risk levels, and a Streamlit interface for non-technical users.

## Features

- CSV input with `text` and `label` columns
- TF-IDF unigram and bigram features
- Numeric content features such as word count, sentence length, punctuation, keyword counts, and simple sentiment score
- Logistic Regression, Naive Bayes, Support Vector Machine, and Perceptron classifiers
- Train-all workflow with model comparison
- Hyperparameter tuning with grid search when the dataset has enough examples
- Adjustable train/test split
- Accuracy, precision, recall, F1-score, ROC-AUC, cross-validation metrics, and confusion matrix output
- Ensemble voting model saved alongside individual classifiers
- Dataset quality warnings for duplicates, short text, and class imbalance
- Binary prediction with probability and Low/Medium/High risk level
- Feature importance chart for interpretable linear models
- Article-level explanation of which active features influenced a prediction
- Batch CSV prediction with downloadable reports
- Downloadable model comparison report
- Suspicious keyword highlighting for pasted text

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train From The Command Line

```powershell
python train.py --data sample_news.csv
```

Use your own CSV by replacing `sample_news.csv`. The file must contain:

- `text`: article, headline, or post content
- `label`: `real` or `fake`, or numeric labels where `0 = real` and `1 = fake`

## Run The Streamlit App

```powershell
streamlit run app.py
```

Then train a model from the sidebar and paste text into the analyzer.

The app trains all classifiers together, displays comparison metrics, saves each model, and uses the best model by default. You can also select a specific trained model for prediction.

## Batch Prediction

Upload a CSV containing a `text` column in the Batch Prediction panel. The app returns:

- predicted label
- fake probability
- real probability
- risk level

The prediction table can be downloaded as a CSV report.

## Project Structure

```text
app.py              Streamlit user interface
train.py            Command-line model training
sample_news.csv     Tiny example dataset
src/features.py     Text cleaning, content features, risk level
src/model.py        Training, prediction, model persistence, feature importance
requirements.txt    Python dependencies
```
