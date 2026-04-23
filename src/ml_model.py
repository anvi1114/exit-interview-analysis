import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def prepare_features(df: pd.DataFrame):
    """Prepare X and y from cleaned dataframe."""
    df = df.copy()

    # Encode any remaining object columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    if 'Attrition' not in df.columns:
        raise ValueError("'Attrition' column not found.")

    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    return X, y


def train_model(X, y, n_estimators=100, random_state=42):
    """Train Random Forest classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test) -> dict:
    """Return accuracy, classification report, confusion matrix."""
    y_pred = model.predict(X_test)
    return {
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'y_pred': y_pred,
        'y_test': y_test
    }


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Return top feature importances as DataFrame."""
    fi = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    return fi


def save_model(model, path: str = 'models/rf_model.pkl'):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str = 'models/rf_model.pkl'):
    """Load model from disk."""
    return joblib.load(path)
