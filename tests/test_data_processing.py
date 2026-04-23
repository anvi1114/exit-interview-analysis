import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_processing import (
    clean_data, encode_features, add_derived_features,
    get_attrition_rate, get_attrition_by_department
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Age': [25, 35, 45, 28, 50],
        'Attrition': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Department': ['Sales', 'HR', 'Sales', 'R&D', 'HR'],
        'MonthlyIncome': [3000, 7000, 5000, 9000, 4000],
        'YearsAtCompany': [1, 6, 3, 11, 2],
        'EmployeeCount': [1, 1, 1, 1, 1],
        'EmployeeNumber': [1, 2, 3, 4, 5],
        'Over18': ['Y', 'Y', 'Y', 'Y', 'Y'],
        'StandardHours': [80, 80, 80, 80, 80],
        'OverTime': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'JobRole': ['Sales Exec', 'HR Rep', 'Manager', 'Scientist', 'Director'],
        'BusinessTravel': ['Travel_Rarely'] * 5,
        'EducationField': ['Life Sciences'] * 5,
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'MaritalStatus': ['Single', 'Married', 'Divorced', 'Single', 'Married'],
    })


class TestCleanData:
    def test_drops_unnecessary_columns(self, sample_df):
        cleaned = clean_data(sample_df)
        for col in ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']:
            assert col not in cleaned.columns

    def test_removes_duplicates(self, sample_df):
        df_with_dupes = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
        cleaned = clean_data(df_with_dupes)
        assert len(cleaned) == len(sample_df)

    def test_shape_after_cleaning(self, sample_df):
        cleaned = clean_data(sample_df)
        assert cleaned.shape[0] == 5
        assert 'Age' in cleaned.columns

    def test_no_null_values(self, sample_df):
        cleaned = clean_data(sample_df)
        assert cleaned.isnull().sum().sum() == 0


class TestEncodeFeatures:
    def test_no_object_columns_after_encoding(self, sample_df):
        cleaned = clean_data(sample_df)
        encoded = encode_features(cleaned)
        object_cols = encoded.select_dtypes(include='object').columns.tolist()
        assert len(object_cols) == 0

    def test_numeric_columns_preserved(self, sample_df):
        cleaned = clean_data(sample_df)
        encoded = encode_features(cleaned)
        assert encoded['Age'].dtype in [np.int64, np.float64]
        assert encoded['MonthlyIncome'].dtype in [np.int64, np.float64]

    def test_shape_unchanged(self, sample_df):
        cleaned = clean_data(sample_df)
        encoded = encode_features(cleaned)
        assert encoded.shape == cleaned.shape


class TestDerivedFeatures:
    def test_tenure_bucket_created(self, sample_df):
        result = add_derived_features(sample_df)
        assert 'TenureBucket' in result.columns

    def test_income_band_created(self, sample_df):
        result = add_derived_features(sample_df)
        assert 'IncomeBand' in result.columns

    def test_tenure_bucket_values(self, sample_df):
        result = add_derived_features(sample_df)
        valid = {'0-2 yrs', '3-5 yrs', '6-10 yrs', '10+ yrs'}
        for val in result['TenureBucket']:
            assert val in valid

    def test_income_band_values(self, sample_df):
        result = add_derived_features(sample_df)
        valid = {'Low', 'Mid', 'High'}
        for val in result['IncomeBand'].dropna():
            assert val in valid


class TestAttritionRate:
    def test_correct_rate(self, sample_df):
        rate = get_attrition_rate(sample_df)
        assert rate == 60.0  # 3 out of 5 = 60%

    def test_returns_float(self, sample_df):
        rate = get_attrition_rate(sample_df)
        assert isinstance(rate, float)

    def test_missing_column_raises(self, sample_df):
        df_no_attr = sample_df.drop(columns=['Attrition'])
        with pytest.raises(ValueError):
            get_attrition_rate(df_no_attr)


class TestAttritionByDept:
    def test_returns_dataframe(self, sample_df):
        result = get_attrition_by_department(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, sample_df):
        result = get_attrition_by_department(sample_df)
        assert 'Department' in result.columns
        assert 'Attrition_Rate' in result.columns

    def test_all_depts_present(self, sample_df):
        result = get_attrition_by_department(sample_df)
        assert set(result['Department']) == {'Sales', 'HR', 'R&D'}
