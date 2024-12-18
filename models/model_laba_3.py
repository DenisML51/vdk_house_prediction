import pandas as pd
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def get_data():
    data = pd.read_csv('C:/Coding/vdk_prices - Copy/Diamonds_for_Sarah.csv')

    return data

def plot_distributions(data):
    plt.figure(figsize=(10, 6))
    sns.displot(data['Price'], kde=True)
    plt.title('Распределение: Price (чистый)')
    price_path = os.path.join("static", "price_distribution.png")
    plt.savefig(price_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.displot(np.log(data['Price']), kde=True)
    plt.title('Распределение: Log(Price)')
    log_price_path = os.path.join("static", "log_price_distribution.png")
    plt.savefig(log_price_path)
    plt.close()

    return price_path, log_price_path

def preprocess(data):
    X = data.drop(['Price', 'ID'], axis=1)
    y = data['Price']
    y = np.log(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def linear(X_train, X_test, y_train, y_test):
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    numerical_features = X_train.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
    numerical_features = [feat for feat in numerical_features if feat not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = pd.DataFrame(
        {
            "MSE": mean_squared_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "Explained Variance": explained_variance_score(y_test, y_pred),
            "MAPE": mean_absolute_error(y_test, y_pred) * 100
        }
    )

    return metrics

