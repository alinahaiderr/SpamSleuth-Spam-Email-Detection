# 📧 SpamSleuth: Statistical and Tree-Based Email Classification

This repository presents a comprehensive machine learning pipeline for detecting spam emails using the **Spambase dataset**. It combines **exploratory data analysis**, **statistical modeling**, and **tree-based classifiers** to uncover patterns and build robust predictive models.

---

## 📦 Dataset Overview

The [Spambase dataset](https://archive.ics.uci.edu/ml/datasets/spambase) contains **4,601 email samples** with **57 features** extracted from email content:
- **Word frequencies**
- **Character frequencies**
- **Capital letter run lengths**
- **Target variable**: `spam` (1 = spam, 0 = non-spam)

---

## 🔍 Exploratory Data Analysis

### 📊 Class Distribution
- Spam: ~39.4%
- Non-Spam: ~60.6%

### 📈 Feature Insights
- Histograms and boxplots reveal distinct distributions of word and character frequencies between spam and non-spam emails.
- Features like `word_freq_free`, `char_freq_$`, and `capital_run_length_longest` show strong separation.

### 🔥 Correlation Analysis
- Spearman correlation heatmaps identify highly correlated features.
- Features like `word_freq_857`, `word_freq_telnet`, and `word_freq_85` were excluded to reduce multicollinearity.

---

## 🧠 Statistical Modeling

### 📌 Logistic Regression (StatsModels)
- Built using statistically significant features (p-value < 0.05).
- Formula-based modeling for interpretability.
- Identified key predictors of spam with strong statistical backing.

---

## 🌳 Tree-Based Classification

### 🌲 Decision Tree Classifier
- Trained with `max_depth=15` and `ccp_alpha=0.001`:
  - Accuracy: ~92%
  - Balanced precision and recall
- Also tested with `max_depth=3` for interpretability.

### 🔍 Hyperparameter Tuning
- GridSearchCV used to optimize:
  - `max_depth`
  - `ccp_alpha`
  - `min_samples_split`
  - `min_samples_leaf`
- Best Decision Tree:
  - Depth: 15
  - Accuracy: ~93%

---

## 🌲🌲 Random Forest Classifier

### 🧪 Grid Search Optimization
- Parameters tuned:
  - `max_depth`, `min_samples_split`, `min_samples_leaf`, `ccp_alpha`
- Best Random Forest:
  - Accuracy: **95.2%**
  - Robust to overfitting
  - Handles feature interactions effectively

---

## 📊 Evaluation Metrics

All models were evaluated using:
- **Confusion Matrix**
- **Classification Report**:
  - Precision, Recall, F1-Score
- **Cross-Validation Accuracy**

Visualizations included for:
- Confusion matrix
- Classification report metrics (bar plots)

---

## 🛠️ Installation

Install required packages:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn statsmodels
