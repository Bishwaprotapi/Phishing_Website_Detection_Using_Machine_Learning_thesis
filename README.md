# Phishing Website Detection Using Machine Learning

A comparative machine learning framework for detecting phishing websites from URL-based features. This repository includes a publication-ready Jupyter notebook with baseline models, hyperparameter tuning, ensemble voting, and professional evaluation visualizations.

## Features

- **10 baseline classifiers** — Logistic Regression, Decision Tree, KNN, SVM, Random Forest, Gradient Boosting, XGBoost, AdaBoost, Naive Bayes, and MLP
- **Hyperparameter tuning** — `GridSearchCV` (Random Forest) and `RandomizedSearchCV` (XGBoost, SVM)
- **Ensemble methods** — Hard voting and soft voting classifiers with tuned estimators
- **Evaluation metrics** — Accuracy, Precision, Recall, F1, 10-fold CV mean, and standard deviation
- **Visual analytics** — ROC curves with AUC, confusion matrices, learning curves, feature importance (Top 10 + % contribution), optional SHAP plots
- **Auto-generated summary** — `best_model_summary.json` after the final conclusion cell runs

## Dataset

| File | Description |
|------|-------------|
| `phishing.csv` | URL feature dataset with a binary `class` label (legitimate vs phishing) |

Place `phishing.csv` in the project root before running the notebook.

## Requirements

- Python 3.10 or newer
- Dependencies listed in [`requirements.txt`](requirements.txt)

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Phishing_Website_Detection_Using_Machine_Learning_thesis.git
cd Phishing_Website_Detection_Using_Machine_Learning_thesis

# Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Open the main notebook:

   ```bash
   jupyter notebook Phishing_Website_Detection_Using_Machine_Learning_New.ipynb
   ```

2. Run cells **top to bottom** (Kernel → Restart & Run All), or follow this order for the publication sections:

   | Step | Section |
   |------|---------|
   | 1 | Import libraries, load `phishing.csv` |
   | 2 | EDA (missing values, distributions, correlation) |
   | 3 | Train/test split and feature scaling |
   | 4 | Baseline model training and results table |
   | 5 | Cross-validation, hard/soft voting |
   | 6 | Confusion matrices, ROC, feature importance, learning curve |
   | 7 | **Research Gap** → hyperparameter tuning → metrics table → ROC (AUC) → SHAP → learning curves → conclusion |

3. After the conclusion cell completes, check the output file:

   ```text
   best_model_summary.json
   ```

   Example structure:

   ```json
   {
     "Best model": "Soft Voting (Tuned)",
     "Best accuracy": 0.97422,
     "Why best": "...",
     "Practical application": "...",
     "Future scope": "..."
   }
   ```

## Project Structure

```text
.
├── Phishing_Website_Detection_Using_Machine_Learning_New.ipynb  # Main notebook
├── phishing.csv                                                  # Dataset
├── requirements.txt                                              # Python dependencies
├── best_model_summary.json                                       # Generated after run
├── README.md
└── LICENSE
```

## Notebook Highlights (Publication Sections)

- **Research Gap** — Motivation and contribution for journal-style writing
- **Hyperparameter tuning** — RF (grid), XGBoost & SVM (randomized search)
- **Professional plots** — Consistent palette, grid, higher DPI, AUC on ROC curves
- **Hard vs soft voting** — Side-by-side comparison table
- **Learning curves** — Random Forest, XGBoost, and SVM
- **Strong conclusion** — Best model, accuracy, rationale, applications, and future work

## Optional: SHAP

SHAP plots are optional. If `shap` is not installed, the notebook skips SHAP and continues. Install it with:

```bash
pip install shap
```

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Bishwaprotap Ray
