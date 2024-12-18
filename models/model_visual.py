import seaborn as sns
import stats
import scipy.stats as stats
import joblib

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split

import seaborn as sns
import scipy.stats as stats
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


def configure_dark_theme():
    plt.style.use('dark_background')

    plt.rcParams.update({
        'figure.facecolor': '#1f2026',
        'axes.facecolor': '#1f2026',
        'savefig.facecolor': '#1f2026',
        'grid.color': 'gray',
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'axes.labelcolor': 'white',
        'axes.edgecolor': 'gray',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'text.color': 'white',
        'lines.color': 'white',
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'legend.facecolor': 'gray'
    })

def linear_reg_visual(data_w, save_path):
    configure_dark_theme()

    X = data_w.drop('price', axis=1)
    y = data_w['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = 'C:\\Coding\\vdk_prices\\ml\\linear_regression_model.pkl'
    pipeline = joblib.load(model_path)

    y_pred = pipeline.predict(X_test)


    plt.scatter(y_test, y_pred, color='blue', edgecolor='white', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
    plt.title('Фактические значения против предсказанных', color='white')
    plt.xlabel('Фактические значения', color='white')
    plt.ylabel('Предсказанные значения', color='white')
    plt.grid(True, color='gray')
    plt.savefig(os.path.join(save_path, 'linear_reg_1.png'))
    plt.close()

    residuals = y_test - y_pred


    plt.scatter(y_pred, residuals, color='purple', edgecolor='white', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.title('Остатки по предсказанным значениям', color='white')
    plt.xlabel('Предсказанные значения', color='white')
    plt.ylabel('Остатки', color='white')
    plt.grid(True, color='gray')
    plt.savefig(os.path.join(save_path, 'linear_reg_2.png'))
    plt.close()


    sns.histplot(residuals, kde=True, color='green')
    plt.title('Распределение остатков', color='white')
    plt.xlabel('Остатки', color='white')
    plt.ylabel('Частота', color='white')
    plt.grid(True, color='gray')
    plt.savefig(os.path.join(save_path, 'linear_reg_3.png'))
    plt.close()


    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot: Проверка нормальности остатков', color='white')
    plt.grid(True, color='gray')
    plt.savefig(os.path.join(save_path, 'linear_reg_4.png'))
    plt.close()

    return True

def model_visualizations(data_w, model_type, save_path):
    X = data_w.drop('price', axis=1)
    y = data_w['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = f'C:\\Coding\\vdk_prices\\ml\\{model_type}_regression_model.pkl'
    pipeline = joblib.load(model_path)

    y_pred = pipeline.predict(X_test)


    plt.scatter(y_test, y_pred, color='blue', edgecolor='white', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
    plt.title(f'Фактические против предсказанных значений: {model_type}')
    plt.xlabel('Фактические значения')
    plt.ylabel('Предсказанные значения')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'{model_type}_1.png'))
    plt.close()

    residuals = y_test - y_pred


    plt.scatter(y_pred, residuals, color='purple', edgecolor='white', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.title(f'Остатки по предсказанным значениям: {model_type}')
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Остатки')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'{model_type}_2.png'))
    plt.close()


    sns.histplot(residuals, kde=True, color='green')
    plt.title(f'Распределение остатков: {model_type}')
    plt.xlabel('Остатки')
    plt.ylabel('Частота')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'{model_type}_3.png'))
    plt.close()


    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot: Проверка нормальности остатков для {model_type}')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'{model_type}_4.png'))
    plt.close()

    return True

def lasso_regression_visual(data_w, save_path):
    return model_visualizations(data_w, 'lasso', save_path)

def ridge_regression_visual(data_w, save_path):
    return model_visualizations(data_w, 'ridge', save_path)

def elastic_net_visual(data_w, save_path):
    return model_visualizations(data_w, 'elastic', save_path)

def random_forest_visual(data_w, save_path):
    X = data_w.drop('price', axis=1)
    y = data_w['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = 'C:\\Coding\\vdk_prices\\ml\\forest_regression_model.pkl'
    rf_search = joblib.load(model_path)

    y_pred = rf_search.predict(X_test)


    plt.scatter(y_test, y_pred, color='blue', edgecolor='white', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Фактические значения против предсказанных для Random Forest', color='white')
    plt.xlabel('Фактические значения', color='white')
    plt.ylabel('Предсказанные значения', color='white')
    plt.grid(True, color='gray')
    plt.savefig(os.path.join(save_path, 'random_forest_1.png'))
    plt.close()

    residuals = y_test - y_pred


    sns.kdeplot(residuals, color='red')
    plt.title('Плотность ошибок для Random Forest', color='white')
    plt.xlabel('Остатки', color='white')
    plt.ylabel('Плотность', color='white')
    plt.grid(True, color='gray')
    plt.savefig(os.path.join(save_path, 'random_forest_2.png'))
    plt.close()


    sns.histplot(y_pred, kde=True, color='blue')
    plt.title('Распределение предсказанных значений для Random Forest', color='white')
    plt.xlabel('Предсказанные значения', color='white')
    plt.ylabel('Частота', color='white')
    plt.grid(True, color='gray')
    plt.savefig(os.path.join(save_path, 'random_forest_3.png'))
    plt.close()


    sns.boxplot(data=[y_test, y_pred], palette=['blue', 'green'])
    plt.title('Boxplot для фактических и предсказанных значений', color='white')
    plt.xticks([0, 1], ['Фактические значения', 'Предсказанные значения'], color='white')
    plt.grid(True, color='gray')
    plt.savefig(os.path.join(save_path, 'random_forest_4.png'))
    plt.close()

    return True

def sv_regression_visual(data_w, save_path):
    X = data_w.drop('price', axis=1)
    y = data_w['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = 'C:\\Coding\\vdk_prices\\ml\\sv_regression_model.pkl'
    pipeline = joblib.load(model_path)

    y_pred = pipeline.predict(X_test)


    plt.scatter(y_test, y_pred, color='blue', edgecolor='white', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
    plt.title('Фактические значения против предсказанных для SVR')
    plt.xlabel('Фактические значения')
    plt.ylabel('Предсказанные значения')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'svr_1.png'))
    plt.close()

    residuals = y_test - y_pred


    plt.scatter(y_pred, residuals, color='purple', edgecolor='white', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.title('Остатки по предсказанным значениям для SVR')
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Остатки')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'svr_2.png'))
    plt.close()


    sns.kdeplot(residuals, color='green')
    plt.title('Плотность ошибок для SVR')
    plt.xlabel('Остатки')
    plt.ylabel('Плотность')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'svr_3.png'))
    plt.close()


    sns.histplot(residuals, kde=True, color='green')
    plt.title('Распределение остатков для SVR')
    plt.xlabel('Остатки')
    plt.ylabel('Частота')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'svr_4.png'))
    plt.close()

    return True
