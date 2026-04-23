import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.ml_model import (
    prepare_features, train_model, evaluate_model, get_feature_importance
)


@pytest.fixture
def model_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        'Age': np.random.randint(22, 60, n),
        'MonthlyIncome': np.random.randint(2000, 15000, n),
        'YearsAtCompany': np.random.randint(0, 20, n),
        'JobSatisfaction': np.random.randint(1, 5, n),
        'WorkLifeBalance': np.random.randint(1, 4, n),
        'OverTime': np.random.randint(0, 2, n),
        'DistanceFromHome': np.random.randint(1, 30, n),
        'Attrition': np.random.randint(0, 2, n),
    })


class TestPrepareFeatures:
    def test_returns_X_and_y(self, model_df):
        X, y = prepare_features(model_df)
        assert X is not None
        assert y is not None

    def test_attrition_not_in_X(self, model_df):
        X, y = prepare_features(model_df)
        assert 'Attrition' not in X.columns

    def test_y_is_attrition(self, model_df):
        X, y = prepare_features(model_df)
        assert list(y) == list(model_df['Attrition'])

    def test_missing_attrition_raises(self, model_df):
        df_no_attr = model_df.drop(columns=['Attrition'])
        with pytest.raises(ValueError):
            prepare_features(df_no_attr)


class TestTrainModel:
    def test_model_trains(self, model_df):
        X, y = prepare_features(model_df)
        model, X_train, X_test, y_train, y_test = train_model(X, y)
        assert model is not None

    def test_train_test_split_ratio(self, model_df):
        X, y = prepare_features(model_df)
        model, X_train, X_test, y_train, y_test = train_model(X, y)
        total = len(X_train) + len(X_test)
        assert total == len(X)
        assert abs(len(X_test) / total - 0.2) < 0.05

    def test_model_has_predict(self, model_df):
        X, y = prepare_features(model_df)
        model, X_train, X_test, y_train, y_test = train_model(X, y)
        assert hasattr(model, 'predict')


class TestEvaluateModel:
    def test_returns_dict(self, model_df):
        X, y = prepare_features(model_df)
        model, X_train, X_test, y_train, y_test = train_model(X, y)
        result = evaluate_model(model, X_test, y_test)
        assert isinstance(result, dict)

    def test_accuracy_in_range(self, model_df):
        X, y = prepare_features(model_df)
        model, X_train, X_test, y_train, y_test = train_model(X, y)
        result = evaluate_model(model, X_test, y_test)
        assert 0 <= result['accuracy'] <= 100

    def test_confusion_matrix_shape(self, model_df):
        X, y = prepare_features(model_df)
        model, X_train, X_test, y_train, y_test = train_model(X, y)
        result = evaluate_model(model, X_test, y_test)
        cm = result['confusion_matrix']
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_required_keys_present(self, model_df):
        X, y = prepare_features(model_df)
        model, X_train, X_test, y_train, y_test = train_model(X, y)
        result = evaluate_model(model, X_test, y_test)
        for key in ['accuracy', 'classification_report', 'confusion_matrix']:
            assert key in result


class TestFeatureImportance:
    def test_returns_dataframe(self, model_df):
        X, y = prepare_features(model_df)
        model, X_train, X_test, y_train, y_test = train_model(X, y)
        fi = get_feature_importance(model, list(X.columns))
        assert isinstance(fi, pd.DataFrame)

    def test_correct_columns(self, model_df):
        X, y = prepare_features(model_df)
        model, X_train, X_test, y_train, y_test = train_model(X, y)
        fi = get_feature_importance(model, list(X.columns))
        assert 'Feature' in fi.columns
        assert 'Importance' in fi.columns

    def test_sorted_descending(self, model_df):
        X, y = prepare_features(model_df)
        model, X_train, X_test, y_train, y_test = train_model(X, y)
        fi = get_feature_importance(model, list(X.columns))
        importances = fi['Importance'].tolist()
        assert importances == sorted(importances, reverse=True)
