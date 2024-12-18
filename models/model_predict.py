import joblib
import pandas as pd
import numpy as np


def predict_price_with_model(user_data, model_name):
    model_paths = {
        'linear_regression': 'C:\\Coding\\vdk_prices\\ml\\linear_regression_model.pkl',
        'lasso_regression': 'C:\\Coding\\vdk_prices\\ml\\lasso_regression_model.pkl',
        'ridge_regression': 'C:\\Coding\\vdk_prices\\ml\\ridge_regression_model.pkl',
        'elastic_net': 'C:\\Coding\\vdk_prices\\ml\\elastic_regression_model.pkl',
        'random_forest': 'C:\\Coding\\vdk_prices\\ml\\forest_regression_model.pkl',
        'sv_regression': 'C:\\Coding\\vdk_prices\\ml\\sv_regression_model.pkl'
    }

    if model_name not in model_paths:
        raise ValueError("Модель с указанным названием не найдена.")

    model_path = model_paths[model_name]
    pipeline = joblib.load(model_path)

    data_df = pd.DataFrame([user_data])
    columns_to_remove = ['price', 'Unnamed: 0']
    data_df.drop(columns=[col for col in columns_to_remove if col in data_df.columns], inplace=True)


    prediction = pipeline.predict(data_df)

    return np.exp(prediction[0])
