from flask import render_template, request, session
from pandas import DataFrame

from app import app
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import os
import joblib

modelParams = {
    "linear_regression": {
        "fit_intercept": {
            "type": "bool",
            "description": "Добавлять ли константу в модель (1 - да, 0 - нет)."
        },
        "normalize": {
            "type": "bool",
            "description": "Применять ли нормализацию признаков (1 - да, 0 - нет)."
        }
    },
    "lasso_regression": {
        "alpha": {
            "type": "float",
            "description": "Параметр регуляризации, контролирующий величину штрафа (0.0 < alpha ≤ 1.0)."
        },
        "fit_intercept": {
            "type": "bool",
            "description": "Добавлять ли константу в модель (1 - да, 0 - нет)."
        },
        "max_iter": {
            "type": "integer",
            "description": "Максимальное количество итераций для оптимизации (≥ 1)."
        },
        "tol": {
            "type": "float",
            "description": "Точность остановки алгоритма (tol > 0)."
        }
    },
    "ridge_regression": {
        "alpha": {
            "type": "float",
            "description": "Параметр регуляризации, контролирующий величину штрафа (0.0 < alpha ≤ 1.0)."
        },
        "fit_intercept": {
            "type": "bool",
            "description": "Добавлять ли константу в модель (1 - да, 0 - нет)."
        },
        "solver": {
            "type": "select:svd,cholesky,lsqr,sparse_cg,sag,saga,auto",
            "description": "Метод решения для оптимизации."
        },
        "max_iter": {
            "type": "integer",
            "description": "Максимальное количество итераций для оптимизации (≥ 1)."
        }
    },
    "random_forest": {
        "n_estimators": {
            "type": "integer",
            "description": "Количество деревьев в лесу (n_estimators ≥ 1)."
        },
        "max_depth": {
            "type": "integer",
            "description": "Максимальная глубина деревьев (max_depth ≥ 1 или None)."
        },
        "min_samples_split": {
            "type": "integer",
            "description": "Минимальное количество выборок для разбиения узла (≥ 2)."
        },
        "min_samples_leaf": {
            "type": "integer",
            "description": "Минимальное количество выборок в листовом узле (≥ 1)."
        },
        "max_features": {
            "type": "select:auto,sqrt,log2",
            "description": "Количество признаков для поиска лучшего разделения."
        },
        "bootstrap": {
            "type": "bool",
            "description": "Использовать ли выборку с возвращением (1 - да, 0 - нет)."
        }
    },
    "svr": {
        "C": {
            "type": "float",
            "description": "Параметр регуляризации (C > 0)."
        },
        "kernel": {
            "type": "select:linear,poly,rbf,sigmoid",
            "description": "Тип ядра, используемого алгоритмом."
        },
        "gamma": {
            "type": "select:scale,auto",
            "description": "Коэффициент для ядра."
        },
        "epsilon": {
            "type": "float",
            "description": "Допустимая ошибка в рамках модели (ε ≥ 0)."
        },
        "degree": {
            "type": "integer",
            "description": "Степень полинома для ядра poly (degree ≥ 1)."
        }
    },
    "Bayesian_Ridge": {
        "alpha_1": {
            "type": "float",
            "description": "Гиперпараметр для априорного распределения альфа 1 (alpha_1 > 0)."
        },
        "alpha_2": {
            "type": "float",
            "description": "Гиперпараметр для априорного распределения альфа 2 (alpha_2 > 0)."
        },
        "lambda_1": {
            "type": "float",
            "description": "Гиперпараметр для априорного распределения лямбда 1 (lambda_1 > 0)."
        },
        "lambda_2": {
            "type": "float",
            "description": "Гиперпараметр для априорного распределения лямбда 2 (lambda_2 > 0)."
        },
        "tol": {
            "type": "float",
            "description": "Точность остановки алгоритма (tol > 0)."
        }
    },
    "SGD_Regressor": {
        "alpha": {
            "type": "float",
            "description": "Параметр регуляризации (alpha > 0)."
        },
        "max_iter": {
            "type": "integer",
            "description": "Максимальное количество итераций (≥ 1)."
        },
        "tol": {
            "type": "float",
            "description": "Точность остановки алгоритма (tol > 0)."
        },
        "learning_rate": {
            "type": "select:constant,optimal,invscaling,adaptive",
            "description": "Метод изменения скорости обучения."
        },
        "eta0": {
            "type": "float",
            "description": "Начальная скорость обучения (eta0 > 0)."
        }
    },
    "Gradient_Boosting_Regressor": {
        "n_estimators": {
            "type": "integer",
            "description": "Количество базовых моделей (деревьев) (n_estimators ≥ 1)."
        },
        "max_depth": {
            "type": "integer",
            "description": "Максимальная глубина дерева (max_depth ≥ 1)."
        },
        "learning_rate": {
            "type": "float",
            "description": "Скорость обучения (0.0 < learning_rate ≤ 1.0)."
        },
        "loss": {
            "type": "select:squared_error,absolute_error,huber,quantile",
            "description": "Функция потерь для оптимизации."
        },
        "subsample": {
            "type": "float",
            "description": "Доля выборки для построения каждого дерева (0.0 < subsample ≤ 1.0)."
        },
        "min_samples_split": {
            "type": "integer",
            "description": "Минимальное количество выборок для разбиения узла (≥ 2)."
        },
        "min_samples_leaf": {
            "type": "integer",
            "description": "Минимальное количество выборок в листовом узле (≥ 1)."
        }
    }
};

def format_price(price):
    return f"{price:,.0f}".replace(",", " ")

def get_data():
    df = pd.read_csv('Diamonds_for_Sarah.csv')

    data = df.copy()

    X = data.drop(['Price', 'ID'], axis=1)
    y = data['Price']
    y = np.log(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
    numerical_features = [feat for feat in numerical_features if feat not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    return preprocessor, X_train, X_test, y_train, y_test


def train_model(model, X_train, X_test, y_train, y_test, preprocessor):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)

    metrics = pd.DataFrame(
        {
            "MSE - test": mean_squared_error(y_test, y_pred),
            "MAE - test": mean_absolute_error(y_test, y_pred),
            "MAPE - test": f'{mean_absolute_error(y_test, y_pred) * 100:.3f}%',
            "R2 - test": r2_score(y_test, y_pred),
        }, index=[0]
    )

    metrics_train = pd.DataFrame(
        {
            "MSE - train": mean_squared_error(y_pred_train, y_train),
            'MAE - train': mean_absolute_error(y_train, y_pred_train),
            'MAPE - train': f'{mean_absolute_error(y_train, y_pred_train) * 100:.3f}%',
            'R2 - train': r2_score(y_train, y_pred_train)
        }, index=[0]
    )
    return metrics, metrics_train, pipeline


MODEL_MAPPING = {
    "linear_regression": LinearRegression(),
    "lasso_regression": Lasso(alpha=0.1),
    "ridge_regression": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    "svr": SVR(C=1.0, kernel='rbf'),
    "Bayesian_Ridge": BayesianRidge(alpha_1=0.1, alpha_2=0.2, ),
    'Gradient_Boosting_Regressor': GradientBoostingRegressor(n_estimators=100, max_depth=10, random_state=42),
    "SGD_Regressor": SGDRegressor(alpha=0.1, max_iter=1000)
}


@app.route('/diamonds', methods=['GET', 'POST'])
def diamonds():
    qwert = 1
    if session['prediction'] is not None:
        session['prediction'] = session['prediction']

    if request.method == 'POST' and request.form.get('model1') == "1":
        model_name = request.form['model']
        model = MODEL_MAPPING.get(model_name)

        hyperparams = {}

        for key, value in request.form.items():
            if key not in ["model", "Carat Weight", "Cut", "Color", "Clarity", "Polish", "Symmetry", "Report", "model1"]:
                param_details = modelParams.get(model_name, {}).get(key)
                if not param_details:
                    return f"Гиперпараметр {key} не поддерживается для модели {model_name}", 400

                param_type = param_details['type']
                if param_type == "integer":
                    hyperparams[key] = int(value)
                elif param_type == "float":
                    hyperparams[key] = float(value)
                elif param_type == 'bool':
                    hyperparams[key] = bool(int(value))
                elif param_type.startswith("select"):
                    if value not in param_type.split(":")[1].split(","):
                        return f"Недопустимое значение {value} для параметра {key}", 400
                    hyperparams[key] = value
                else:
                    hyperparams[key] = value

        model.set_params(**hyperparams)
        preprocessor, X_train, X_test, y_train, y_test = get_data()
        metrics, metrics_train, pipeline = train_model(model, X_train, X_test, y_train, y_test, preprocessor)
        metrics = metrics.to_html(classes="metrics-table", index=False, border=0)
        metrics_train = metrics_train.to_html(classes="metrics-table", index=False, border=0)
        session['metrics'] = metrics
        session['metrics_train'] = metrics_train
        model_path = 'C:\\Coding\\vdk_prices\\ml\\diamond_model.pkl'

        if os.path.exists(model_path):
            os.remove(model_path)
            joblib.dump(pipeline, model_path)
            print('Модель переобучена')
        else:
            joblib.dump(pipeline, model_path)

    if request.method == 'post' and request.form.get('data') == "1":
        input_data = pd.DataFrame({
            'Carat Weight': [float(request.form['Carat Weight'])],
            'Cut': [request.form['Cut']],
            'Color': [request.form['Color']],
            'Clarity': [request.form['Clarity']],
            'Polish': [request.form['Polish']],
            'Symmetry': [request.form['Symmetry']],
            'Report': [request.form['Report']]
        })

        pipeline = joblib.load('C:\\Coding\\vdk_prices\\ml\\diamond_model.pkl')
        qwert = 1234
        prediction = format_price(np.exp(pipeline.predict(input_data)[0]))
        session['prediction'] = prediction

        return render_template(
            'diamonds.html',
            models=MODEL_MAPPING.keys(),
            metrics=session['metrics'],
            metrics_train=session['metrics_train'],
            prediction=prediction,
            qwert=qwert)

    return render_template(
        'diamonds.html',
        models=MODEL_MAPPING.keys(),
        metrics=session['metrics'],
        metrics_train = session['metrics_train'],
        prediction=session['prediction'],
        qwert=qwert)


