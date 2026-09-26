import pytest
import numpy as np
import pandas as pd
from templates.return_data import RegressionResults, ClassiferResults
from src.models import ModelTrainer

class TestModelTrainerRegression:
    def test_train_linear_returns_regression_results(self, sample_regression_data):
        signs, target = sample_regression_data
        model = ModelTrainer()
        res = model.train_linear(signs, target, mode='linear')
        assert isinstance(res, RegressionResults)

    def test_train_linear_r2_is_valid(self, sample_regression_data):
        """R² должен быть числом (может быть отрицательным)."""
        signs, target = sample_regression_data
        model = ModelTrainer()
        res = model.train_linear(signs, target, mode='linear')
        assert isinstance(res.r2, float)
        assert not np.isnan(res.r2)

    def test_train_linear_mae_positive(self, sample_regression_data):
        """MAE всегда ≥ 0."""
        signs, target = sample_regression_data
        model = ModelTrainer()
        res = model.train_linear(signs, target, mode='linear')
        assert res.mae >= 0

    def test_train_linear_mse_positive(self, sample_regression_data):
        """MSE всегда ≥ 0."""
        signs, target = sample_regression_data
        model = ModelTrainer()
        res = model.train_linear(signs, target, mode='linear')
        assert res.mse >= 0

    def test_train_linear_coef_length(self, sample_regression_data):
        """Длина коэффициентов = число признаков после get_dummies."""
        signs, target = sample_regression_data
        model = ModelTrainer()
        res = model.train_linear(signs, target, mode='linear')
        assert len(res.k) == len(res.signs_column)

    @pytest.mark.parametrize("mode", ['linear', 'ridge', 'lasso', 'random', 'gradient'])
    def test_train_linear_all_modes(self, sample_regression_data, mode):
        """Все режимы регрессии должны работать."""
        signs, target = sample_regression_data
        model = ModelTrainer()
        res = model.train_linear(signs, target, mode=mode)
        assert res.r2 is not None

    def test_train_linear_invalid_mode_fallback(self, sample_regression_data):
        """Неизвестный режим должен откатиться к линейной регрессии."""
        signs, target = sample_regression_data
        model = ModelTrainer()
        res = model.train_linear(signs, target, mode='unknown')
        assert isinstance(res, RegressionResults)

class TestModelTrainerClassifier:
    """Тесты для классификатора."""

    def test_train_classifier_returns_classifier_results(self, sample_classification_data):
        """Проверяем тип результата."""
        signs, target = sample_classification_data
        model = ModelTrainer()
        res = model.train_classifier(signs, target, mode='logistic')
        assert isinstance(res, ClassiferResults)

    def test_train_classifier_accuracy_range(self, sample_classification_data):
        """Accuracy должен быть в [0, 1]."""
        signs, target = sample_classification_data
        model = ModelTrainer()
        res = model.train_classifier(signs, target, mode='logistic')
        assert 0 <= res.accuracy <= 1

    def test_train_classifier_confusion_matrix_shape(self, sample_classification_data):
        """Confusion matrix должна быть 2x2 для бинарной классификации."""
        signs, target = sample_classification_data
        model = ModelTrainer()
        res = model.train_classifier(signs, target, mode='logistic')
        assert res.matrix.shape == (2, 2)

    def test_train_classifier_report_keys(self, sample_classification_data):
        """В отчёте должны быть accuracy, precision, recall, f1-score."""
        signs, target = sample_classification_data
        model = ModelTrainer()
        res = model.train_classifier(signs, target, mode='logistic')
        assert 'accuracy' in res.report
        assert 'macro avg' in res.report

    @pytest.mark.parametrize("mode", ['logistic', 'random', 'histgradient'])
    def test_train_classifier_all_modes(self, sample_classification_data, mode):
        """Все режимы классификатора должны работать."""
        signs, target = sample_classification_data
        model = ModelTrainer()
        res = model.train_classifier(signs, target, mode=mode)
        assert res.accuracy > 0


class TestScaleCoef:
    """Тесты масштабирования."""

    def test_scale_coef_returns_array(self, sample_regression_data):
        """scale_coef возвращает numpy-массив."""
        signs, _ = sample_regression_data
        model = ModelTrainer()
        result = model.scale_coef(signs.select_dtypes(include=[np.number]))
        assert isinstance(result, np.ndarray)

    def test_scale_coef_mean_near_zero(self, sample_regression_data):
        """После StandardScaler среднее ≈ 0."""
        signs, _ = sample_regression_data
        signs_num = signs.select_dtypes(include=[np.number])
        model = ModelTrainer()
        result = model.scale_coef(signs_num)
        assert np.allclose(result.mean(axis=0), 0, atol=1e-7)


class TestPredictClassifier:
    """Тесты предсказания."""

    def test_predict_classifier_returns_array(self, sample_classification_data):
        """predict_classifier возвращает массив предсказаний."""
        signs, target = sample_classification_data
        model = ModelTrainer()
        model.train_classifier(signs, target, mode='logistic')
        # Загружаем модель из кэша
        import joblib
        trained = joblib.load('models/classifier-logistic-model.pkl')
        from src.data_loader import Loader
        loader = Loader()
        signs_encoded = loader.encode_categorical(signs)
        signs_encoded = signs_encoded.reindex(columns=trained.feature_names_in_, fill_value=0)
        preds = ModelTrainer.predict_classifier(trained, signs_encoded)
        assert len(preds) == len(signs)