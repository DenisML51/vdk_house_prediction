import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score


def linear_metrics(data_w):
    X = data_w.drop('price', axis=1)
    y = data_w['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = 'C:\\Coding\\vdk_prices\\ml\\linear_regression_model.pkl'
    pipeline = joblib.load(model_path)

    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    metrics_df = pd.DataFrame({
        'MSE': [mse],
        'R2': [r2],
        'MAE': [mae],
        'Explained Variance': [evs]
    })

    return metrics_df

def lasso_metrics(data_w):
    X = data_w.drop('price', axis=1)
    y = data_w['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = 'C:\\Coding\\vdk_prices\\ml\\lasso_regression_model.pkl'
    pipeline = joblib.load(model_path)

    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    metrics_df = pd.DataFrame({
        'MSE': [mse],
        'R2': [r2],
        'MAE': [mae],
        'Explained Variance': [evs]
    })

    return metrics_df

def ridge_metrics(data_w):
    X = data_w.drop('price', axis=1)
    y = data_w['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = 'C:\\Coding\\vdk_prices\\ml\\ridge_regression_model.pkl'
    pipeline = joblib.load(model_path)

    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    metrics_df = pd.DataFrame({
        'MSE': [mse],
        'R2': [r2],
        'MAE': [mae],
        'Explained Variance': [evs]
    })

    return metrics_df

def elastic_metrics(data_w):
    X = data_w.drop('price', axis=1)
    y = data_w['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = 'C:\\Coding\\vdk_prices\\ml\\elastic_regression_model.pkl'
    pipeline = joblib.load(model_path)

    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    metrics_df = pd.DataFrame({
        'MSE': [mse],
        'R2': [r2],
        'MAE': [mae],
        'Explained Variance': [evs]
    })

    return metrics_df

def forest_metrics(data_w):
    X = data_w.drop('price', axis=1)
    y = data_w['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = 'C:\\Coding\\vdk_prices\\ml\\forest_regression_model.pkl'
    pipeline = joblib.load(model_path)

    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    metrics_df = pd.DataFrame({
        'MSE': [mse],
        'R2': [r2],
        'MAE': [mae],
        'Explained Variance': [evs]
    })

    return metrics_df

def svr_metrics(data_w):
    X = data_w.drop('price', axis=1)
    y = data_w['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = 'C:\\Coding\\vdk_prices\\ml\\sv_regression_model.pkl'
    pipeline = joblib.load(model_path)

    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    metrics_df = pd.DataFrame({
        'MSE': [mse],
        'R2': [r2],
        'MAE': [mae],
        'Explained Variance': [evs]
    })

    return metrics_df
