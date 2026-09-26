# tests/conftest.py
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_regression_data():
    """Маленький датасет для тестов регрессии."""
    np.random.seed(42)
    n = 50
    signs = pd.DataFrame({
        'age': np.random.randint(18, 60, n),
        'bedtime_screen_time_minutes': np.random.uniform(0, 200, n),
        'doomscroll_sessions_per_night': np.random.randint(0, 10, n),
        'keeps_phone_in_bedroom': np.random.choice(['Yes', 'No'], n),
    })
    target = pd.DataFrame({
        'sleep_hours_per_night': np.random.uniform(4, 10, n)
    })
    return signs, target


@pytest.fixture
def sample_classification_data():
    """Маленький датасет для тестов классификации."""
    np.random.seed(42)
    n = 100
    signs = pd.DataFrame({
        'age': np.random.randint(18, 60, n),
        'bedtime_screen_time_minutes': np.random.uniform(0, 200, n),
        'doomscroll_sessions_per_night': np.random.randint(0, 10, n),
        'keeps_phone_in_bedroom': np.random.choice(['Yes', 'No'], n),
    })
    target = pd.DataFrame({
        'doomscroller': np.random.choice(['Yes', 'No'], n)
    })
    return signs, target


@pytest.fixture
def sample_csv(tmp_path):
    """Создаёт временный CSV для тестов Loader."""
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'occupation_status': ['Student', 'Employed', 'Student', 'Employed', 'Student'],
        'country_region': ['USA', 'UK', 'USA', 'UK', 'USA'],
        'primary_device_used_at_night': ['Phone', 'Laptop', 'Phone', 'Tablet', 'Phone'],
        'uses_night_mode': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'keeps_phone_in_bedroom': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'consumes_negative_news_content': ['No', 'Yes', 'No', 'Yes', 'No'],
        'uses_sleep_tracking_app': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'doomscroller': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'sleep_hours_per_night': [7.0, 8.0, 6.5, 7.5, 6.0],
        'bedtime_screen_time_minutes': [50, 30, 80, 20, 60],
    })
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)