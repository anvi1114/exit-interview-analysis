# Exit Interview Data Analysis — DA50

> **HR Analytics | Data Analytics + NLP Dashboard | Production Project**

A production-grade data analytics project that analyzes employee exit interview data to uncover attrition patterns, sentiment trends, and actionable HR insights. Built with Python, deployed on Streamlit Cloud.

---

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## Project Structure

```
exit-interview-analysis/
│
├── app.py                  ← Streamlit dashboard (5 pages)
├── requirements.txt        ← Python dependencies
├── README.md
├── .gitignore
│
├── src/
│   ├── data_processing.py  ← Load, clean, encode, feature engineering
│   ├── ml_model.py         ← Random Forest classifier + evaluation
│   ├── nlp_analysis.py     ← TextBlob sentiment analysis
│   └── visualization.py    ← All matplotlib/seaborn charts
│
├── tests/
│   ├── test_data_processing.py
│   ├── test_ml_model.py
│   └── test_nlp_analysis.py
│
└── data/
    └── WA_Fn-UseC_-HR-Employee-Attrition.csv
```

---

## Features

- **Overview Dashboard** — KPI cards, attrition rate, dataset preview
- **EDA** — 6 charts: dept attrition, income, tenure, age, overtime, correlation heatmap
- **NLP Sentiment** — TextBlob analysis on 20 exit feedback responses, word frequency
- **ML Model** — Random Forest with 88% accuracy, confusion matrix, feature importance
- **Report** — Executive summary, recommendations, downloadable report

---

## Dataset

**IBM HR Analytics Employee Attrition Dataset**
- Source: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- 1,470 employees · 35 columns · Target: `Attrition` (Yes/No)

---

## Installation & Local Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/exit-interview-analysis.git
cd exit-interview-analysis

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add dataset
# Place WA_Fn-UseC_-HR-Employee-Attrition.csv in the data/ folder

# 5. Run app
streamlit run app.py
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as main file
4. Click **Deploy** — live in 2 minutes!

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualizations |
| TextBlob | NLP sentiment analysis |
| Scikit-learn | Random Forest ML model |
| Streamlit | Web dashboard |
| Pytest | Unit testing |

---

## Key Findings

- **16% attrition rate** — above industry benchmark
- **Top 3 drivers:** MonthlyIncome, OverTime, Age
- **ML Model accuracy:** 88%
- **Sentiment:** 60% of exit feedback is negative (management & salary themes)

---

*DA50 · HR Analytics Group Project*
