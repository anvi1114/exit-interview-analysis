import pandas as pd
from textblob import TextBlob
from collections import Counter
import re
import string

SAMPLE_FEEDBACK = [
    {"EmployeeID": 1, "Department": "Sales", "Feedback": "Management was poor and targets were unrealistic. No support from seniors."},
    {"EmployeeID": 2, "Department": "HR", "Feedback": "Great team and culture. Leaving for better salary elsewhere."},
    {"EmployeeID": 3, "Department": "R&D", "Feedback": "Loved the work but work-life balance was terrible. Too much overtime."},
    {"EmployeeID": 4, "Department": "Sales", "Feedback": "No growth opportunities. Same role for 3 years with no promotion."},
    {"EmployeeID": 5, "Department": "HR", "Feedback": "Toxic workplace environment. Management does not listen to employees."},
    {"EmployeeID": 6, "Department": "R&D", "Feedback": "Excellent projects and learning opportunities. Personal reasons for leaving."},
    {"EmployeeID": 7, "Department": "Sales", "Feedback": "Salary was below market rate. Benefits were also not competitive."},
    {"EmployeeID": 8, "Department": "HR", "Feedback": "Good experience overall. Relocating to another city."},
    {"EmployeeID": 9, "Department": "R&D", "Feedback": "Lack of recognition for hard work. Management plays favourites."},
    {"EmployeeID": 10, "Department": "Sales", "Feedback": "High pressure environment. Constant micromanagement made it unbearable."},
    {"EmployeeID": 11, "Department": "HR", "Feedback": "Flexible hours and good benefits but low pay was the main issue."},
    {"EmployeeID": 12, "Department": "R&D", "Feedback": "Amazing colleagues and interesting work. Got a better offer abroad."},
    {"EmployeeID": 13, "Department": "Sales", "Feedback": "Poor communication from leadership. Goals changed every month."},
    {"EmployeeID": 14, "Department": "HR", "Feedback": "Company culture was positive but no career progression available."},
    {"EmployeeID": 15, "Department": "R&D", "Feedback": "Burnout from long hours. No support for mental health or wellbeing."},
    {"EmployeeID": 16, "Department": "Sales", "Feedback": "Commission structure was unfair. Hard work was not rewarded properly."},
    {"EmployeeID": 17, "Department": "HR", "Feedback": "Overall decent experience. Found a role closer to home."},
    {"EmployeeID": 18, "Department": "R&D", "Feedback": "Outdated tools and processes. Felt like innovation was discouraged."},
    {"EmployeeID": 19, "Department": "Sales", "Feedback": "Team was great but manager was very difficult to work with."},
    {"EmployeeID": 20, "Department": "HR", "Feedback": "Enjoyed working here. Pursuing higher education full time now."},
]


def get_sentiment(text: str) -> dict:
    """Return polarity, subjectivity, and label for a text."""
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity, 3)
    if polarity > 0.1:
        label = 'Positive'
    elif polarity < -0.1:
        label = 'Negative'
    else:
        label = 'Neutral'
    return {'polarity': polarity, 'subjectivity': subjectivity, 'label': label}


def analyze_feedback(feedback_list: list = None) -> pd.DataFrame:
    """Run sentiment analysis on feedback list."""
    if feedback_list is None:
        feedback_list = SAMPLE_FEEDBACK
    records = []
    for item in feedback_list:
        sentiment = get_sentiment(item['Feedback'])
        records.append({
            'EmployeeID': item['EmployeeID'],
            'Department': item['Department'],
            'Feedback': item['Feedback'],
            'Polarity': sentiment['polarity'],
            'Subjectivity': sentiment['subjectivity'],
            'Sentiment_Label': sentiment['label']
        })
    return pd.DataFrame(records)


def get_top_negative_words(df: pd.DataFrame, top_n: int = 5) -> list:
    """Return top N words from negative feedback."""
    negative_text = ' '.join(df[df['Sentiment_Label'] == 'Negative']['Feedback'].tolist())
    negative_text = negative_text.lower()
    negative_text = re.sub(r'[^a-z\s]', '', negative_text)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'was', 'is', 'it', 'from', 'very', 'too',
        'not', 'no', 'be', 'been', 'by', 'are', 'were', 'that', 'this',
        'had', 'has', 'have', 'also', 'felt', 'made', 'did', 'my', 'me'
    }
    words = [w for w in negative_text.split() if w not in stop_words and len(w) > 2]
    return Counter(words).most_common(top_n)


def get_negative_feedback_text(df: pd.DataFrame) -> str:
    """Return all negative feedback as single string for wordcloud."""
    return ' '.join(df[df['Sentiment_Label'] == 'Negative']['Feedback'].tolist())
