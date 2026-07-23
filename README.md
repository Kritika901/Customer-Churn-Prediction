# 📊 Customer Churn Prediction

A production-quality, end-to-end machine learning project that predicts
customer churn for a telecom company using the **IBM Telco Customer
Churn** dataset. Includes a full ML pipeline (cleaning → feature
engineering → training → evaluation) and an interactive **Streamlit**
dashboard for exploration and real-time predictions.

---

## 🚀 Project Overview

Customer churn — when a customer stops doing business with a company —
is one of the most expensive problems in subscription-based businesses.
This project builds a complete pipeline to:

1. Clean and prepare the raw Telco dataset
2. Explore the data visually (static + interactive charts)
3. Engineer features (encoding, scaling)
4. Train and compare **Logistic Regression**, **Decision Tree**, and
   **Random Forest** models
5. Automatically select the best-performing model
6. Evaluate it thoroughly (accuracy, precision, recall, F1, ROC-AUC,
   confusion matrix, feature importance)
7. Serve predictions through a multi-page **Streamlit** dashboard,
   supporting both single-customer and batch (CSV) predictions

---

## ✨ Features

- ✅ Modular, type-hinted, documented Python code (`src/`)
- ✅ Automated data cleaning (missing values, duplicates, dtype fixes)
- ✅ Rich EDA — churn distribution, gender/contract analysis, tenure &
  monthly-charge analysis, correlation heatmap, pairplots, interactive
  Plotly charts
- ✅ Feature engineering — label encoding, one-hot encoding, scaling,
  stratified train/test split
- ✅ Multi-model training with automatic best-model selection
- ✅ Full evaluation suite — metrics, ROC curve, confusion matrix,
  feature importance plots (saved to `reports/figures/`)
- ✅ Business insights & retention recommendations
- ✅ Model persistence via `joblib`
- ✅ Streamlit dashboard: Home, About Dataset, EDA, Prediction, Analytics
- ✅ Single-customer prediction with probability, confidence, and
  top contributing factors
- ✅ Batch prediction via CSV upload + downloadable results
- ✅ Logging throughout the pipeline
- ✅ Clean project structure, ready for GitHub

---

## 🗂️ Folder Structure

```
Customer-Churn-Prediction/
│
├── data/
│   ├── raw/                  # Place Telco-Customer-Churn.csv here
│   └── processed/            # Cleaned data is saved here
│
├── notebooks/
│   └── EDA.ipynb             # Exploratory Data Analysis notebook
│
├── src/
│   ├── preprocessing.py      # Data cleaning
│   ├── feature_engineering.py# Encoding, scaling, splitting
│   ├── train.py               # Model training + selection
│   ├── evaluate.py            # Metrics & evaluation plots
│   ├── predict.py             # Inference (single + batch)
│   └── utils.py                # Shared paths, logging, IO helpers
│
├── models/
│   └── churn_model.pkl        # Saved best model (generated)
│
├── reports/
│   └── figures/               # Generated evaluation plots
│
├── streamlit_app/
│   └── app.py                 # Streamlit dashboard
│
├── screenshots/                # App screenshots for README
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| ML | Scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| App | Streamlit |
| Persistence | Joblib |
| Versioning | Git |

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📦 Dataset

This project uses the **IBM Telco Customer Churn** dataset:

🔗 https://www.kaggle.com/datasets/blastchar/telco-customer-churn

1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle.
2. Rename it to `Telco-Customer-Churn.csv`.
3. Place it inside `data/raw/`.

---

## ▶️ Usage

```bash
# 1. Train the model (cleans data, engineers features, trains & evaluates
#    Logistic Regression / Decision Tree / Random Forest, saves the best one)
python src/train.py

# 2. Launch the interactive dashboard
streamlit run streamlit_app/app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

### Optional: Run inference directly from Python

```python
from src.predict import ChurnPredictor

predictor = ChurnPredictor()
result = predictor.predict_single({
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 90.5, "TotalCharges": 452.5,
})
print(result)
```

---

## 📈 Results

After running `python src/train.py`, a comparison report is written to
`reports/metrics.json` and evaluation plots are saved to
`reports/figures/`. Typical performance on this dataset:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.80 | ~0.65 | ~0.55 | ~0.60 | ~0.84 |
| Decision Tree | ~0.78 | ~0.60 | ~0.52 | ~0.56 | ~0.79 |
| Random Forest | ~0.80 | ~0.66 | ~0.51 | ~0.58 | ~0.84 |

*(Exact numbers depend on the dataset version and random seed — the
pipeline automatically selects the best model by ROC-AUC.)*

---

## 🖼️ Screenshots

> Add your own screenshots to the `screenshots/` folder and reference them below.

| Page | Preview |
|---|---|
| Home | `screenshots/home.png` |
| EDA | `screenshots/eda.png` |
| Prediction | `screenshots/prediction.png` |
| Analytics | `screenshots/analytics.png` |

---

## 💡 Business Insights

- Month-to-month contract customers churn significantly more than
  annual/biennial contract customers.
- Fiber optic internet customers, especially without security/support
  add-ons, are higher-risk.
- Electronic check payment users churn more than automatic payment users.
- New customers (low tenure) are the highest-risk segment — early
  onboarding matters.

**Retention recommendations:** incentivize longer contracts, bundle
security/support add-ons at a discount, promote automatic payments, and
run proactive onboarding campaigns for new customers. See the
**Analytics** page in the dashboard for more detail.

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning (GridSearchCV / Optuna)
- [ ] Add gradient boosting models (XGBoost, LightGBM, CatBoost)
- [ ] SHAP-based explainability for individual predictions
- [ ] Model monitoring & drift detection
- [ ] CI/CD pipeline (GitHub Actions) with automated tests
- [ ] Dockerize the app for easier deployment
- [ ] Deploy to Streamlit Community Cloud / AWS / GCP
- [ ] Add authentication for the dashboard

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE)
for details.
