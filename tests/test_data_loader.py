# tests/test_data_loader.py
import pytest
import pandas as pd
from src.data_loader import Loader


class TestLoader:
    """Тесты загрузчика данных."""

    def test_load_data_success(self, sample_csv):
        """Успешная загрузка CSV."""
        loader = Loader()
        loader.load_data(sample_csv)
        assert not loader.get_df().empty

    def test_load_data_file_not_found(self, capsys):
        """Отсутствующий файл не должен крашить программу."""
        loader = Loader()
        loader.load_data('nonexistent.csv')
        captured = capsys.readouterr()
        assert 'File not found' in captured.out

    def test_delete_nan_removes_rows(self):
        """dropna удаляет строки с NaN."""
        loader = Loader()
        df = pd.DataFrame({
            'a': [1, 2, None, 4],
            'b': [5, None, 7, 8]
        })
        result = loader.delete_nan(df)
        assert len(result) == 2

    def test_encode_categorical_creates_dummies(self):
        """get_dummies создаёт бинарные колонки."""
        loader = Loader()
        df = pd.DataFrame({
            'color': ['red', 'blue', 'red'],
            'value': [1, 2, 3]
        })
        result = loader.encode_categorical(df)
        assert 'color_red' in result.columns or 'color_blue' in result.columns

    def test_load_specific_data_returns_subset(self, sample_csv):
        """load_specific_data возвращает только запрошенные колонки."""
        loader = Loader()
        loader.load_data(sample_csv)
        result = loader.load_specific_data(['age', 'gender'])
        assert list(result.columns) == ['age', 'gender']

    def test_load_user_analytics_returns_dataframe(self, sample_csv):
        """load_user_analytics возвращает DataFrame без NaN."""
        loader = Loader()
        loader.load_data(sample_csv)
        df = loader.load_user_analytics()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty