import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw HR attrition dataset."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns and remove duplicates."""
    cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all categorical columns."""
    df = df.copy()
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add TenureBucket and IncomeBand columns."""
    df = df.copy()

    def tenure_bucket(years):
        if years <= 2:
            return '0-2 yrs'
        elif years <= 5:
            return '3-5 yrs'
        elif years <= 10:
            return '6-10 yrs'
        else:
            return '10+ yrs'

    if 'YearsAtCompany' in df.columns:
        df['TenureBucket'] = df['YearsAtCompany'].apply(tenure_bucket)

    if 'MonthlyIncome' in df.columns:
        df['IncomeBand'] = pd.qcut(
            df['MonthlyIncome'], q=3,
            labels=['Low', 'Mid', 'High'],
            duplicates='drop'
        )
    return df


def get_attrition_rate(df: pd.DataFrame) -> float:
    """Return overall attrition rate as percentage."""
    if 'Attrition' not in df.columns:
        raise ValueError("Column 'Attrition' not found.")
    col = df['Attrition']
    return round((col.astype(str) == 'Yes').mean() * 100, 2)


def get_attrition_by_department(df: pd.DataFrame) -> pd.DataFrame:
    """Return attrition rate per department."""
    if 'Department' not in df.columns or 'Attrition' not in df.columns:
        raise ValueError("Required columns missing.")
    df2 = df.copy()
    df2['_attr_flag'] = (df2['Attrition'].astype(str) == 'Yes').astype(int)
    result = df2.groupby('Department')['_attr_flag'].apply(
        lambda x: round(x.mean() * 100, 2)
    ).reset_index()
    result.columns = ['Department', 'Attrition_Rate']
    return result
