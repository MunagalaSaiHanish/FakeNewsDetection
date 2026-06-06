# 📰 Fake News Detection System

> An AI-powered Machine Learning and Natural Language Processing (NLP) application that detects fake news articles using TF-IDF feature engineering, advanced text analytics, and multiple machine learning classifiers.

---

# 📌 Overview

The **Fake News Detection System** is an end-to-end Machine Learning and Natural Language Processing (NLP) solution designed to classify news articles as **Real** or **Fake**.

The system leverages advanced text preprocessing, TF-IDF vectorization, engineered content features, and multiple machine learning algorithms to analyze news content and provide accurate predictions. An interactive Streamlit dashboard enables users to perform single-article analysis, batch predictions, model training, and performance evaluation.

The project was developed to address the growing challenge of misinformation by providing an intelligent and explainable fake news detection platform.

---

# 🎯 Key Highlights

-  Built an end-to-end NLP classification pipeline
-  Implemented and compared multiple Machine Learning algorithms
-  Developed an Ensemble Voting Classifier
-  Achieved **84.77% Accuracy** using Support Vector Machine (Best Model)
-  Performed Hyperparameter Tuning using Grid Search
-  Applied Cross Validation for robust model evaluation
-  Built an interactive Streamlit Web Application
-  Added Explainable AI features for prediction interpretation
-  Enabled Batch Prediction using CSV uploads

---

# 🚀 Features

## 🔍 News Classification

- Real vs Fake News Detection
- Single Article Prediction
- Batch CSV Prediction
- Real-Time Analysis

## 🧠 Natural Language Processing

- Text Cleaning
- Stopword Removal
- Tokenization
- TF-IDF Vectorization
- Unigram & Bigram Features

## 🤖 Machine Learning Models

- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- Perceptron
- Ensemble Voting Classifier

## 📊 Explainability & Analytics

- Confidence Score Estimation
- Risk Level Classification
- Influential Feature Analysis
- Article-Level Explanations
- Confusion Matrix Visualization
- ROC Curve Analysis
- Model Comparison Dashboard

## 💻 User Interface

- Interactive Streamlit Dashboard
- Dataset Preview
- Model Selection
- Training Configuration Panel
- Downloadable Model Comparison Reports

---

# 🏗️ System Architecture

```text
News Article
      │
      ▼
Text Preprocessing
      │
      ▼
Feature Engineering
(TF-IDF + Content Features)
      │
      ▼
Machine Learning Models
(Logistic Regression / Naive Bayes /
 SVM / Perceptron)
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Model Evaluation
      │
      ▼
Prediction & Explainability
      │
      ▼
Streamlit Dashboard
```

---

# 🛠️ Technology Stack

### Programming Language
- Python

### Machine Learning
- Scikit-Learn

### Natural Language Processing
- TF-IDF Vectorization
- N-Gram Features

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib

### Web Application
- Streamlit

### Model Persistence
- Joblib

---

# 📂 Dataset Information

The project was trained using publicly available fake news datasets containing labeled news articles.

### Dataset Statistics

| Metric | Value |
|----------|----------|
| Total Articles | 23,196 |
| Duplicate Articles Detected | 1,472 |
| Very Short Articles Detected | 1,307 |
| Classes | Real & Fake |

The system automatically performs dataset validation and quality analysis before model training.

---

# 🤖 Models Implemented

| Model | Purpose |
|---------|---------|
| Logistic Regression | Baseline Text Classification |
| Naive Bayes | Probabilistic Text Classification |
| Support Vector Machine | High-Performance Linear Classification |
| Perceptron | Neural-Inspired Classification |
| Ensemble Voting | Combined Prediction Strategy |

---

# 📊 Results

## Model Performance Comparison

| Model | Accuracy |
|---------|---------|
| 🏆 Support Vector Machine | **84.77%** |
| Naive Bayes | 84.24% |
| Logistic Regression | 82.12% |
| Perceptron | 76.55% |

---

## Best Model: Support Vector Machine (SVM)

### Cross Validation Performance

| Metric | Value |
|----------|----------|
| Accuracy | 84.66% |
| Precision | 80.82% |
| Recall | 75.71% |
| F1 Score | 78.06% |
| Cross Validation Accuracy | 84.66% ± 0.16% |

---

### Classification Report

| Class | Precision | Recall | F1 Score |
|---------|---------|---------|---------|
| Fake News | 0.87 | 0.94 | 0.90 |
| Real News | 0.75 | 0.58 | 0.65 |

The Support Vector Machine classifier achieved the highest overall performance and was selected as the final deployment model.

---

# 📈 Sample Prediction Output

The system provides:

- Predicted Class (Real/Fake)
- Confidence Score
- Risk Level Assessment
- Influential Features
- Article-Level Explanation
- Class Probability Distribution

Example:

```text
Classification: Fake

Fake Probability: 74.6%

Risk Level: Medium
```

---

# 📁 Project Structure

```text
fake-news-detection-system/
│
├── app.py
├── train.py
├── models/
├── data/
├── src/
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/MunagalaSaiHanish/fake-news-detection-system.git
cd fake-news-detection-system
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The Streamlit application will launch locally in your browser.

---

# 🏋️ Model Training

The application supports training multiple machine learning models directly through the Streamlit interface.

### Steps

1. Launch the application:

```bash
streamlit run app.py
```

2. Upload a dataset containing the required columns:

```text
text,label
```

3. Configure:
   - Test Set Size
   - Hyperparameter Tuning
   - Model Selection

4. Click **Train All Classifiers**.

### Training Output

The system automatically:

- Preprocesses and cleans text data
- Generates TF-IDF features
- Trains Logistic Regression, Naive Bayes, SVM, and Perceptron models
- Performs Hyperparameter Tuning
- Evaluates all models using Accuracy, Precision, Recall, F1-Score, and Cross Validation
- Selects the best-performing model
- Saves trained models for future predictions

> Training may take several minutes depending on dataset size and system configuration.


# 🔮 Future Enhancements

- BERT-Based Fake News Detection
- Transformer Models
- Multilingual Classification
- Real-Time News Verification APIs
- News Source Credibility Analysis
- Deep Learning Architectures
- Cloud Deployment

---

# 👨‍💻 Author

## Munagala Sai Hanish

B.Tech Computer Science Engineering  
JNTUH College of Engineering

### Connect With Me

- GitHub: https://github.com/MunagalaSaiHanish
- LinkedIn: https://www.linkedin.com/in/sai-hanish-munagala-55846b339
