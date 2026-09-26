# tests/test_visualization.py
import matplotlib
matplotlib.use('Agg')  # Важно! До импорта pyplot

import pytest
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from src.visualization import Graph


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'gender': ['Male', 'Female', 'Male', 'Female'],
        'age': [25, 30, 35, 40],
        'occupation_status': ['Student', 'Employed', 'Student', 'Employed'],
        'country_region': ['USA', 'UK', 'USA', 'UK'],
    })


class TestGraph:
    def test_visualization_user_analytics_returns_figure(self, sample_df):
        """Метод должен вернуть matplotlib Figure."""
        graph = Graph()
        fig = graph.visualization_user_analytics(sample_df)
        assert isinstance(fig, Figure)

    def test_coef_visual_returns_figure(self):
        """coef_visual возвращает Figure."""
        graph = Graph()
        coef = np.array([0.5, -0.3, 0.8])
        names = ['a', 'b', 'c']
        fig = graph.coef_visual(coef, names)
        assert isinstance(fig, Figure)

    def test_heatmap_returns_figure(self):
        """heatmap возвращает Figure."""
        graph = Graph()
        matrix = np.array([[10, 2], [3, 15]])
        fig = graph.heatmap(matrix)
        assert isinstance(fig, Figure)

    def test_prediction_vs_actual_returns_figure(self):
        """prediction_vs_actual возвращает Figure."""
        graph = Graph()
        y_real = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        fig = graph.prediction_vs_actual(y_pred, y_real)
        assert isinstance(fig, Figure)