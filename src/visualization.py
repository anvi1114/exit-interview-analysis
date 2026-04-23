import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import io
import base64


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string for Streamlit."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


def plot_attrition_by_dept(df: pd.DataFrame):
    """Bar chart: attrition rate by department."""
    col = df['Attrition']
    if col.dtype == object:
        dept_attr = df.groupby('Department')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
    else:
        dept_attr = df.groupby('Department')['Attrition'].apply(
            lambda x: x.mean() * 100
        ).reset_index()
    dept_attr.columns = ['Department', 'Attrition_Rate']

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=dept_attr, x='Department', y='Attrition_Rate',
                palette='viridis', ax=ax)
    ax.set_title('Attrition Rate by Department', fontsize=14, fontweight='bold')
    ax.set_ylabel('Attrition Rate (%)')
    ax.set_xlabel('Department')
    fig.tight_layout()
    return fig


def plot_tenure_boxplot(df: pd.DataFrame):
    """Box plot: years at company vs attrition."""
    df2 = df.copy()
    if df2['Attrition'].dtype != object:
        df2['Attrition_Label'] = df2['Attrition'].map({1: 'Left', 0: 'Stayed'})
    else:
        df2['Attrition_Label'] = df2['Attrition'].map({'Yes': 'Left', 'No': 'Stayed'})

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df2, x='Attrition_Label', y='YearsAtCompany',
                palette='Set2', ax=ax)
    ax.set_title('Tenure vs Attrition', fontsize=14, fontweight='bold')
    ax.set_xlabel('Attrition')
    ax.set_ylabel('Years at Company')
    fig.tight_layout()
    return fig


def plot_age_histogram(df: pd.DataFrame):
    """Histogram: age distribution by attrition."""
    df2 = df.copy()
    if df2['Attrition'].dtype != object:
        left = df2[df2['Attrition'] == 1]['Age']
        stayed = df2[df2['Attrition'] == 0]['Age']
    else:
        left = df2[df2['Attrition'] == 'Yes']['Age']
        stayed = df2[df2['Attrition'] == 'No']['Age']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(stayed, bins=20, alpha=0.6, color='#4ecdc4', label='Stayed')
    ax.hist(left, bins=20, alpha=0.6, color='#ff6b6b', label='Left')
    ax.set_title('Age Distribution by Attrition', fontsize=14, fontweight='bold')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    ax.legend()
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    """Heatmap: correlation of numeric columns."""
    numeric_df = df.select_dtypes(include='number')
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm',
                linewidths=0.3, ax=ax, center=0)
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_overtime_attrition(df: pd.DataFrame):
    """Count plot: overtime vs attrition."""
    df2 = df.copy()
    if df2['Attrition'].dtype != object:
        df2['Attrition_Label'] = df2['Attrition'].map({1: 'Left', 0: 'Stayed'})
    else:
        df2['Attrition_Label'] = df2['Attrition'].map({'Yes': 'Left', 'No': 'Stayed'})

    if df2['OverTime'].dtype != object:
        df2['OverTime_Label'] = df2['OverTime'].map({1: 'Yes', 0: 'No'})
    else:
        df2['OverTime_Label'] = df2['OverTime']

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df2, x='OverTime_Label', hue='Attrition_Label',
                  palette='magma', ax=ax)
    ax.set_title('OverTime vs Attrition', fontsize=14, fontweight='bold')
    ax.set_xlabel('OverTime')
    ax.set_ylabel('Count')
    fig.tight_layout()
    return fig


def plot_income_by_dept(df: pd.DataFrame):
    """Bar chart: avg monthly income by dept and attrition."""
    df2 = df.copy()
    if df2['Attrition'].dtype != object:
        df2['Attrition_Label'] = df2['Attrition'].map({1: 'Left', 0: 'Stayed'})
    else:
        df2['Attrition_Label'] = df2['Attrition'].map({'Yes': 'Left', 'No': 'Stayed'})

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df2, x='Department', y='MonthlyIncome',
                hue='Attrition_Label', palette='cool', ax=ax)
    ax.set_title('Avg Monthly Income by Department & Attrition',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Department')
    ax.set_ylabel('Avg Monthly Income')
    fig.tight_layout()
    return fig


def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 10):
    """Horizontal bar: top N feature importances."""
    top = fi_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=top, x='Importance', y='Feature',
                palette='viridis', ax=ax)
    ax.set_title(f'Top {top_n} Attrition Drivers', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: list):
    """Heatmap of confusion matrix."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Stayed', 'Left'],
                yticklabels=['Stayed', 'Left'])
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    fig.tight_layout()
    return fig


def plot_sentiment_by_dept(df: pd.DataFrame):
    """Count plot: sentiment distribution by department."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(data=df, x='Department', hue='Sentiment_Label',
                  palette='viridis', ax=ax)
    ax.set_title('Sentiment Distribution by Department',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Department')
    ax.set_ylabel('Count')
    fig.tight_layout()
    return fig
