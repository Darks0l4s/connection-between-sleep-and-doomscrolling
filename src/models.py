from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import pandas as pd
from numpy import ndarray
import numpy as np
from templates.return_data import RegressionResults, ClassiferResults
import joblib
import os

class ModelTrainer:
    
    def __init__(self):
        pass

    def cash_model(self, trained_models, dataset_type, mode_name):
        folder='models'
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, f'{dataset_type}-{mode_name}-model.pkl')
        joblib.dump(trained_models, file_path)

    def cash_results(self, res, dataset_type, mode_name):
        folder='models'
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, f'{dataset_type}-{mode_name}-res.pkl')
        joblib.dump(res, file_path)

    def train_linear(self, signs: pd.DataFrame, target: ndarray, mode: str) -> RegressionResults:
        if hasattr(target, 'values'):
            target = target.values.ravel()
        else:
            target = np.array(target).ravel()
        match mode:
            case 'linear':
                model=LinearRegression()
            case 'ridge':
                model=Ridge(alpha=10)
            case 'lasso':
                model=Lasso(alpha=0.1)
            case 'random':
                model = RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42, n_jobs=-1)
            case 'gradient':
                model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
            case _:
                model = LinearRegression()
        signs = pd.get_dummies(signs, drop_first=True)
        signs_column = signs.columns
        signs = self.scale_coef(signs)
        X_train, X_test, y_train, y_test = train_test_split(
            signs, target,
            test_size=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred=model.predict(X_test)
        if hasattr(model, 'coef_'):
            k = model.coef_      
            b = model.intercept_ 
        else:
            k = model.feature_importances_
            b = 0.0 
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        y_predict = model.predict(X_train)
        y_real = y_train
        res = RegressionResults(
            signs_column=signs_column,
            k=k, b=b,
            mae=mae, mse=mse, r2=r2,
            y_predict=y_predict, y_real=y_real
        )
        self.cash_model(model, 'regression', mode)
        self.cash_results(res, 'regression', mode)
        return res
    
    def train_classifier(self, signs: pd.DataFrame, target: ndarray, mode: str) -> ClassiferResults:
        if hasattr(target, 'values'):
            target = target.values.ravel()
        signs = pd.get_dummies(signs, drop_first=True)
        match mode:
            case 'logistic':
                model = LogisticRegression(max_iter=1000, random_state=42)
            case 'random':
                model=RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=5,
                    n_jobs=-1,
                    random_state=42
                )
            case 'histgradient':
                model=HistGradientBoostingClassifier(
                    
                    max_iter=100,
                    learning_rate=0.05,
                    max_depth=3,
                    l2_regularization=0.1,
                    random_state=42
                )
            case _:
                model = LogisticRegression(max_iter=1000)
        X_train, X_test, y_train, y_test = train_test_split(
            signs, target, test_size=0.2, stratify=target, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred)
        if hasattr(model, 'coef_'):
            importances = model.coef_[0]
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            result= permutation_importance(model, X_test, y_test, n_repeats=10,random_state=42)
            importances = result.importances_mean
        res = ClassiferResults(
        importances=importances, accuracy=accuracy, report=report, matrix=matrix)
        self.cash_results(res, 'classifier', mode)
        self.cash_model(model, 'classifier', mode)
        return res
    
    def scale_coef(self, X: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(X)
        return x_scaled
    @staticmethod
    def predict_classifier(model, data):
        return model.predict(data)