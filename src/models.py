from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pandas import DataFrame
from numpy import ndarray
from src.data_loader import string_in_int
class LinearML:
    def __init__(self):
        pass

    def train_linear(self, signs: DataFrame, target: DataFrame) -> tuple[DataFrame, ndarray, ndarray, float, float, float]:
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
        return signs_column, k, b, mae, mse, r2

    def scale_coef(self, X: DataFrame) -> DataFrame:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(X)
        return x_scaled