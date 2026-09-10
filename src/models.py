from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from pandas import DataFrame
from numpy import ndarray
from src.data_loader import string_in_int
class LinearML:
    def __init__(self):
        pass

    def train_linear(self, signs: DataFrame, target: DataFrame, mode ='linear') -> tuple[DataFrame, ndarray, ndarray, float, float, float]:

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
        signs = string_in_int(signs)
        signs_column = signs.columns
        signs = self.scale_coef(signs)
        X_train, X_test, y_train, y_test = train_test_split(
            signs, target,
            test_size=0.2
        )
        model.fit(X_train, y_train)
        y_pred=model.predict(X_test)
        k = model.coef_
        b = model.intercept_
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        self.y_predict = model.predict(X_train)
        self.y_real = y_train
        return signs_column, k, b, mae, mse, r2

    def train_classifier(self, signs: DataFrame, target: DataFrame):
        model = LogisticRegression(max_iter=1000)
        X_train, X_test, y_train, y_test = train_test_split(
            signs, target, test_size=0.2, stratify=target
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(X_test, y_pred)
        report = classification_report(X_test, y_pred)
        matrix = confusion_matrix(X_test, y_pred)
        importances = model.coef_[0]
        return importances, accuracy, report, matrix
    
    def scale_coef(self, X: DataFrame) -> DataFrame:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(X)
        return x_scaled