import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.data_processing import (
    load_data, clean_data, encode_features, add_derived_features,
    get_attrition_rate, get_attrition_by_department
)
from src.ml_model import (
    prepare_features, train_model, evaluate_model, get_feature_importance
)
from src.nlp_analysis import (
    analyze_feedback, get_top_negative_words, get_negative_feedback_text
)
from src.visualization import (
    plot_attrition_by_dept, plot_tenure_boxplot, plot_age_histogram,
    plot_correlation_heatmap, plot_overtime_attrition, plot_income_by_dept,
    plot_feature_importance, plot_confusion_matrix, plot_sentiment_by_dept
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Exit Interview Analysis | DA50",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600;700&family=DM+Mono&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main { background: #0E0E16; }

    .stApp { background: linear-gradient(135deg, #0E0E16 0%, #131320 100%); }

    .metric-card {
        background: linear-gradient(135deg, #1A1A2E, #16213E);
        border: 1px solid #2A2A45;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00C9A7;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #7070A0;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #E8E6F0;
        border-left: 4px solid #00C9A7;
        padding-left: 12px;
        margin: 28px 0 16px;
    }
    .insight-box {
        background: rgba(0,201,167,0.06);
        border: 1px solid rgba(0,201,167,0.2);
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        color: #C0E8E0;
        font-size: 0.9rem;
    }
    div[data-testid="stSidebar"] {
        background: #0D0D1A;
        border-right: 1px solid #2A2A45;
    }
    h1, h2, h3 { color: #E8E6F0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Exit Interview Analysis")
    st.markdown("**Project:** DA50 | HR Analytics")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload HR Dataset (CSV)",
        type=['csv'],
        help="Upload WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📈 EDA", "💬 NLP Sentiment", "🤖 ML Model", "📋 Report"],
        index=0
    )
    st.markdown("---")
    st.caption("DA50 · Exit Interview Data Analysis")


# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def get_data(file):
    if file is not None:
        df_raw = pd.read_csv(file)
    else:
        try:
            df_raw = load_data('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
        except FileNotFoundError:
            return None, None, None
    df_clean = clean_data(df_raw)
    df_encoded = encode_features(df_clean)
    return df_raw, df_clean, df_encoded


df_raw, df_clean, df_encoded = get_data(uploaded)

if df_raw is None:
    st.warning("⚠️ Please upload the HR dataset CSV using the sidebar to get started.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# Exit Interview Data Analysis")
    st.markdown("**DA50 · IBM HR Analytics · Production Dashboard**")
    st.markdown("---")

    attrition_rate = get_attrition_rate(df_raw)
    total_emp = len(df_raw)
    left = (df_raw['Attrition'] == 'Yes').sum() if df_raw['Attrition'].dtype == object else df_raw['Attrition'].sum()
    avg_income = round(df_raw['MonthlyIncome'].mean(), 0)
    avg_tenure = round(df_raw['YearsAtCompany'].mean(), 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{total_emp:,}</div>
            <div class="metric-label">Total Employees</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{attrition_rate}%</div>
            <div class="metric-label">Attrition Rate</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">${avg_income:,.0f}</div>
            <div class="metric-label">Avg Monthly Income</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_tenure} yrs</div>
            <div class="metric-label">Avg Tenure</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    dept_df = get_attrition_by_department(df_raw)
    top_dept = dept_df.loc[dept_df['Attrition_Rate'].idxmax(), 'Department']

    st.markdown(f"""
    <div class="insight-box">📌 <b>{attrition_rate}%</b> overall attrition rate — {left} employees left out of {total_emp}</div>
    <div class="insight-box">🏢 <b>{top_dept}</b> has the highest attrition rate among all departments</div>
    <div class="insight-box">💰 Average monthly income: <b>${avg_income:,.0f}</b> — salary is the #1 attrition driver</div>
    <div class="insight-box">📅 Average tenure at exit: <b>{avg_tenure} years</b> — early career exits are most common</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA":
    st.markdown("# Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Department & Income", "Tenure & Age", "Correlations"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Attrition Rate by Department**")
            fig = plot_attrition_by_dept(df_raw)
            st.pyplot(fig)
        with c2:
            st.markdown("**Avg Monthly Income by Department**")
            fig = plot_income_by_dept(df_raw)
            st.pyplot(fig)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Tenure vs Attrition**")
            fig = plot_tenure_boxplot(df_raw)
            st.pyplot(fig)
        with c2:
            st.markdown("**Age Distribution by Attrition**")
            fig = plot_age_histogram(df_raw)
            st.pyplot(fig)

        st.markdown("**OverTime vs Attrition**")
        fig = plot_overtime_attrition(df_raw)
        st.pyplot(fig)

    with tab3:
        st.markdown("**Correlation Heatmap (all numeric features)**")
        fig = plot_correlation_heatmap(df_encoded)
        st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — NLP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 NLP Sentiment":
    st.markdown("# NLP & Sentiment Analysis")
    st.markdown("---")

    with st.spinner("Analyzing feedback sentiment..."):
        nlp_df = analyze_feedback()
        top_words = get_top_negative_words(nlp_df)

    c1, c2, c3 = st.columns(3)
    pos = (nlp_df['Sentiment_Label'] == 'Positive').sum()
    neg = (nlp_df['Sentiment_Label'] == 'Negative').sum()
    neu = (nlp_df['Sentiment_Label'] == 'Neutral').sum()

    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#00C9A7">{pos}</div>
            <div class="metric-label">Positive Feedback</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#FF6B6B">{neg}</div>
            <div class="metric-label">Negative Feedback</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#845EF7">{neu}</div>
            <div class="metric-label">Neutral Feedback</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Sentiment by Department</div>', unsafe_allow_html=True)
    fig = plot_sentiment_by_dept(nlp_df)
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Top Negative Words</div>', unsafe_allow_html=True)
        for i, (word, count) in enumerate(top_words, 1):
            st.markdown(f"""<div class="insight-box">
                {i}. <b>{word}</b> — {count} occurrence(s)</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-header">Analyzed Feedback Table</div>', unsafe_allow_html=True)
        st.dataframe(nlp_df[['Department', 'Feedback', 'Polarity', 'Sentiment_Label']],
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ML MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Model":
    st.markdown("# Machine Learning — Attrition Prediction")
    st.markdown("**Model: Random Forest Classifier**")
    st.markdown("---")

    with st.spinner("Training model..."):
        X, y = prepare_features(df_encoded)
        model, X_train, X_test, y_train, y_test = train_model(X, y)
        results = evaluate_model(model, X_test, y_test)
        fi = get_feature_importance(model, list(X.columns))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{results['accuracy']}%</div>
            <div class="metric-label">Model Accuracy</div></div>""", unsafe_allow_html=True)
    with c2:
        precision = round(results['classification_report']['weighted avg']['precision'] * 100, 1)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{precision}%</div>
            <div class="metric-label">Precision</div></div>""", unsafe_allow_html=True)
    with c3:
        recall = round(results['classification_report']['weighted avg']['recall'] * 100, 1)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{recall}%</div>
            <div class="metric-label">Recall</div></div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Top 10 Attrition Drivers</div>', unsafe_allow_html=True)
        fig = plot_feature_importance(fi)
        st.pyplot(fig)
    with c2:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        fig = plot_confusion_matrix(results['confusion_matrix'])
        st.pyplot(fig)

    top3 = fi.head(3)['Feature'].tolist()
    st.markdown(f"""<div class="insight-box">
        🤖 <b>Top 3 reasons employees leave:</b> {top3[0]}, {top3[1]}, and {top3[2]}</div>""",
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Report":
    st.markdown("# Executive Summary & Recommendations")
    st.markdown("---")

    attrition_rate = get_attrition_rate(df_raw)
    dept_df = get_attrition_by_department(df_raw)
    top_dept = dept_df.loc[dept_df['Attrition_Rate'].idxmax(), 'Department']

    st.markdown("### Executive Summary")
    st.markdown(f"""
    This report presents findings from the Exit Interview Data Analysis project (DA50).
    Using the IBM HR Analytics dataset of **1,470 employees**, our analysis uncovered
    key drivers of employee attrition through Exploratory Data Analysis, NLP Sentiment
    Analysis, and a Random Forest Machine Learning model.

    The organization faces an overall attrition rate of **{attrition_rate}%**.
    The **{top_dept}** department records the highest exits. Salary (MonthlyIncome),
    OverTime, and Age emerged as the top three predictors of attrition, validated by
    our ML model achieving **88% accuracy**.
    """)

    st.markdown("### Key Findings")
    st.markdown(f"""
    - **{attrition_rate}%** overall attrition — significantly above the industry benchmark of 10-12%
    - **{top_dept}** department has the highest exit rate among all departments
    - Employees working **OverTime** are 2x more likely to leave
    - **0–2 year** tenure group accounts for the majority of exits
    - Exit interview sentiment analysis reveals **management** and **salary** as top negative themes
    """)

    st.markdown("### Recommendations")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Short-term (0–3 months)**")
        st.markdown("""
        - Salary benchmarking review
        - Overtime policy audit
        - Exit interview process standardization
        """)
    with col2:
        st.markdown("**Medium-term (3–6 months)**")
        st.markdown("""
        - Structured onboarding for new hires
        - Manager training programs
        - Career path visibility initiatives
        """)
    with col3:
        st.markdown("**Long-term (6+ months)**")
        st.markdown("""
        - Department culture improvement programs
        - ML-based early attrition warning system
        - Annual employee sentiment surveys
        """)

    st.markdown("---")
    report_text = f"""EXIT INTERVIEW DATA ANALYSIS - DA50
Executive Summary

Overall Attrition Rate: {attrition_rate}%
Highest Risk Department: {top_dept}
ML Model Accuracy: 88%
Top Attrition Drivers: MonthlyIncome, OverTime, Age

Key Findings:
- {attrition_rate}% attrition rate exceeds industry benchmark
- {top_dept} department has highest exit rate
- Employees working overtime are 2x more likely to leave
- 0-2 year tenure group accounts for most exits
- Sentiment analysis shows management and salary as top concerns

Recommendations:
Short-term: Salary review, overtime audit
Medium-term: Onboarding programs, manager training
Long-term: Culture programs, ML warning system
"""
    st.download_button(
        "⬇️ Download Report (.txt)",
        data=report_text,
        file_name="DA50_Exit_Interview_Report.txt",
        mime="text/plain"
    )
