import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


def linear_regression(data_w):
    X = data_w.drop('price', axis=1)
    y = data_w['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numerical_features = X.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
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
    linear_model = pipeline.named_steps['model']

    intercept = linear_model.intercept_
    coefficients = linear_model.coef_

    formula_latex = r"\displaystyle y = {:.3f}".format(intercept)
    for i, col in enumerate(X_train.columns):
        coeff = coefficients[i]

        if coeff >= 0:
            formula_latex += r" + ({:.3f} \cdot \text{{{}}})".format(coeff, col)
        else:
            formula_latex += r" - ({:.3f} \cdot \text{{{}}})".format(abs(coeff), col)



    model_path = 'C:\\Coding\\vdk_prices\\ml\\linear_regression_model.pkl'
    if os.path.exists(model_path):
        os.remove(model_path)
        joblib.dump(pipeline, model_path)
        print('Модель переобучена')
    else:
        joblib.dump(pipeline, model_path)



    return True

def lasso_regression(data_w):
    X = data_w.drop('price', axis=1)
    y = data_w['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numerical_features = X.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
    numerical_features = [feat for feat in numerical_features if feat not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', Lasso(alpha=0.1))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

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
    lasso_model = pipeline.named_steps['model']

    intercept = lasso_model.intercept_
    coefficients = lasso_model.coef_

    formula_latex = r"\displaystyle y = {:.3f}".format(intercept)
    for i, col in enumerate(X_train.columns):
        coeff = coefficients[i]

        if coeff >= 0:
            formula_latex += r" + ({:.3f} \cdot \text{{{}}})".format(coeff, col)
        else:
            formula_latex += r" - ({:.3f} \cdot \text{{{}}})".format(abs(coeff), col)

    model_path = 'C:\\Coding\\vdk_prices\\ml\\lasso_regression_model.pkl'
    if os.path.exists(model_path):
        os.remove(model_path)
        joblib.dump(pipeline, model_path)
        print('Модель переобучена')

    else:
        joblib.dump(pipeline, model_path)

    return True

def ridge_regression(data_w):
    X = data_w.drop('price', axis=1)
    y = data_w['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numerical_features = X.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
    numerical_features = [feat for feat in numerical_features if feat not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=0.1))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

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
    ridge_model = pipeline.named_steps['model']

    intercept = ridge_model.intercept_
    coefficients = ridge_model.coef_

    formula_latex = r"\displaystyle y = {:.3f}".format(intercept)
    for i, col in enumerate(X_train.columns):
        coeff = coefficients[i]

        if coeff >= 0:
            formula_latex += r" + ({:.3f} \cdot \text{{{}}})".format(coeff, col)
        else:
            formula_latex += r" - ({:.3f} \cdot \text{{{}}})".format(abs(coeff), col)

    model_path = 'C:\\Coding\\vdk_prices\\ml\\ridge_regression_model.pkl'
    if os.path.exists(model_path):
        os.remove(model_path)
        joblib.dump(pipeline, model_path)
        print('Модель переобучена')

    else:
        joblib.dump(pipeline, model_path)

    return True

def elastic_net(data_w):
    X = data_w.drop('price', axis=1)
    y = data_w['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numerical_features = X.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
    numerical_features = [feat for feat in numerical_features if feat not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', ElasticNet(alpha=0.1, l1_ratio=0.1))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

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
    elasticnet_model = pipeline.named_steps['model']

    intercept = elasticnet_model.intercept_
    coefficients = elasticnet_model.coef_

    formula_latex = r"\displaystyle y = {:.3f}".format(intercept)
    for i, col in enumerate(X_train.columns):
        coeff = coefficients[i]

        if coeff >= 0:
            formula_latex += r" + ({:.3f} \cdot \text{{{}}})".format(coeff, col)
        else:
            formula_latex += r" - ({:.3f} \cdot \text{{{}}})".format(abs(coeff), col)

    model_path = 'C:\\Coding\\vdk_prices\\ml\\elastic_regression_model.pkl'
    if os.path.exists(model_path):
        os.remove(model_path)
        joblib.dump(pipeline, model_path)
        print('Модель переобучена')

    else:
        joblib.dump(pipeline, model_path)

    return True

def random_forest(data_w):
    X = data_w.drop('price', axis=1)
    y = data_w['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numerical_features = X.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
    numerical_features = [feat for feat in numerical_features if feat not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    rf_model = RandomForestRegressor()
    rf_param_grid = {
        'model__n_estimators': [50, 100, 200, 300],
        'model__max_features': ['auto', 'sqrt', 'log2'],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', rf_model)
    ])

    rf_search = RandomizedSearchCV(rf_pipeline, rf_param_grid, n_iter=3, cv=3, verbose=1, random_state=42)
    rf_search.fit(X_train, y_train)

    y_pred = rf_search.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

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

    model_path = 'C:\\Coding\\vdk_prices\\ml\\forest_regression_model.pkl'
    if os.path.exists(model_path):
        os.remove(model_path)
        joblib.dump(rf_search, model_path)
        print('Модель переобучена')

    else:
        joblib.dump(rf_search, model_path)

    return True

def sv_regression(data_w):
    X = data_w.drop('price', axis=1)
    y = data_w['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numerical_features = X.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
    numerical_features = [feat for feat in numerical_features if feat not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', SVR(kernel='rbf', C=1.0, epsilon=0.1))
    ])

    pipeline.fit(X_train, y_train)
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

    model_path = 'C:\\Coding\\vdk_prices\\ml\\sv_regression_model.pkl'
    if os.path.exists(model_path):
        os.remove(model_path)
        joblib.dump(pipeline, model_path)
        print('Модель переобучена')

    else:
        joblib.dump(pipeline, model_path)

    return True

