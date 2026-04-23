import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.nlp_analysis import (
    get_sentiment, analyze_feedback, get_top_negative_words,
    get_negative_feedback_text, SAMPLE_FEEDBACK
)


class TestGetSentiment:
    def test_positive_text(self):
        result = get_sentiment("I loved working here, great team and amazing culture!")
        assert result['label'] == 'Positive'
        assert result['polarity'] > 0.1

    def test_negative_text(self):
        result = get_sentiment("Terrible management and toxic workplace environment.")
        assert result['label'] == 'Negative'
        assert result['polarity'] < -0.1

    def test_neutral_text(self):
        result = get_sentiment("I worked here for two years.")
        assert result['label'] == 'Neutral'

    def test_returns_dict(self):
        result = get_sentiment("Some feedback text here.")
        assert isinstance(result, dict)
        assert 'polarity' in result
        assert 'subjectivity' in result
        assert 'label' in result

    def test_polarity_range(self):
        result = get_sentiment("This is a sample text.")
        assert -1.0 <= result['polarity'] <= 1.0

    def test_subjectivity_range(self):
        result = get_sentiment("This is a sample text.")
        assert 0.0 <= result['subjectivity'] <= 1.0


class TestAnalyzeFeedback:
    def test_returns_dataframe(self):
        df = analyze_feedback()
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self):
        df = analyze_feedback()
        for col in ['EmployeeID', 'Department', 'Feedback',
                    'Polarity', 'Subjectivity', 'Sentiment_Label']:
            assert col in df.columns

    def test_row_count(self):
        df = analyze_feedback()
        assert len(df) == len(SAMPLE_FEEDBACK)

    def test_sentiment_labels_valid(self):
        df = analyze_feedback()
        valid = {'Positive', 'Negative', 'Neutral'}
        assert set(df['Sentiment_Label'].unique()).issubset(valid)

    def test_custom_feedback(self):
        custom = [
            {"EmployeeID": 99, "Department": "Test",
             "Feedback": "Great place to work!"}
        ]
        df = analyze_feedback(custom)
        assert len(df) == 1
        assert df.iloc[0]['Sentiment_Label'] == 'Positive'


class TestTopNegativeWords:
    def test_returns_list(self):
        df = analyze_feedback()
        result = get_top_negative_words(df)
        assert isinstance(result, list)

    def test_returns_correct_count(self):
        df = analyze_feedback()
        result = get_top_negative_words(df, top_n=5)
        assert len(result) <= 5

    def test_each_item_is_tuple(self):
        df = analyze_feedback()
        result = get_top_negative_words(df)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2


class TestNegativeFeedbackText:
    def test_returns_string(self):
        df = analyze_feedback()
        result = get_negative_feedback_text(df)
        assert isinstance(result, str)

    def test_not_empty_when_negatives_exist(self):
        df = analyze_feedback()
        result = get_negative_feedback_text(df)
        assert len(result) > 0
