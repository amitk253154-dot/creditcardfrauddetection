# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using [Logistic Regression / Random Forest / XGBoost].

## Problem Statement
Credit card fraud causes large financial losses. This project builds a model that classifies transactions as fraudulent or legitimate. Fraud is rare, so the dataset is highly imbalanced.

## Dataset
- Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 transactions, of which only 492 (~0.17%) are fraud
- Features: `V1–V28` (PCA-transformed), `Time`, `Amount`, `Class` (0 = legit, 1 = fraud)

## Approach
1. Data exploration and visualization
2. Preprocessing (scaling `Amount` and `Time`)
3. Handling class imbalance (SMOTE / undersampling / class weights)
4. Model training (list the models you tried)
5. Evaluation and comparison

## Results
| Model | Precision | Recall | F1-score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | | | | |
| Random Forest | | | | |
| XGBoost | | | | |

Accuracy is not a useful metric here, so I focus on precision, recall, F1, and PR-AUC.

## Project Structure
```
├── data/            # dataset (not included, download from Kaggle)
├── notebooks/       # EDA and experiments
├── src/             # preprocessing, training, evaluation scripts
├── models/          # saved models
├── requirements.txt
└── README.md
```

## Installation
```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

## Usage
```bash
python src/train.py
python src/predict.py --input sample.csv
```

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, imbalanced-learn, XGBoost, Matplotlib, Seaborn

## Future Improvements
- Real-time detection API (Flask / FastAPI)
- Hyperparameter tuning
- Deep learning approaches (autoencoders)

## Authors
Your name(s) – [GitHub/LinkedIn]

## License