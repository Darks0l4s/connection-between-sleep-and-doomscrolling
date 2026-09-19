from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from pandas import DataFrame, get_dummies
from numpy import ndarray

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
        signs = get_dummies(signs, drop_first=True)
        signs_column = signs.columns
        signs = self.scale_coef(signs)
        X_train, X_test, y_train, y_test = train_test_split(
            signs, target,
            test_size=0.1
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
        self.y_predict = model.predict(X_train)
        self.y_real = y_train
        return signs_column, k, b, mae, mse, r2

    def train_classifier(self, signs: DataFrame, target: DataFrame, mode='logistic'):
        match mode:
            case 'logistic':
                model = LogisticRegression(max_iter=1000)
            case 'random':
                model=RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=5,
                    n_jobs=-1
                )
            case 'histgradient':
                model=HistGradientBoostingClassifier(
                    max_iter=100,
                    learning_rate=0.05,
                    max_depth=3,
                    l2_regularization=0.1
                )
            case _:
                model = LogisticRegression(max_iter=1000)
        X_train, X_test, y_train, y_test = train_test_split(
            signs, target, test_size=0.2, stratify=target
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
            importances = [0] * len(signs.columns) 
        return importances, accuracy, report, matrix
    
    def scale_coef(self, X: DataFrame) -> DataFrame:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(X)
        return x_scaled