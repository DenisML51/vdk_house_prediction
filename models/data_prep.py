import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings(action="ignore")


def get_data_prep(url):
    data = pd.read_csv(url)
    data_w = data.copy()
    data_w.columns = data_w.columns.str.replace(' ', '')

    data_w['Жилая'] = data_w['Жилая'].str.replace('м2', '').str.replace(',', '.').astype(float)

    medians = data_w.select_dtypes(include='number').median()
    data_w.fillna(value=medians, inplace=True)

    modes = data_w.select_dtypes(include='object').mode().iloc[0]
    data_w.fillna(value=modes, inplace=True)

    columns_to_remove = ['Unnamed:0', 'Unnamed:0.1']
    data_w.drop(columns=[col for col in columns_to_remove if col in data_w.columns], inplace=True)


    data_w['price'] = np.log1p(data_w['price'])

    return data_w

def cluster(df):
    df = df[df['longitude'] > 100]
    co = {
        'latitude': df['latitude'],
        'longitude': df['longitude']
    }

    array = pd.DataFrame(co).dropna()

    kmeans = KMeans(n_clusters=7, algorithm='lloyd')
    kmeans.fit(array)

    df['cluster'] = kmeans.fit_predict(array)

    df['cluster'] = df['cluster'].astype('object')

    too_much_na_columns = ['latitude', 'longitude']
    df.drop(columns=too_much_na_columns, inplace=True)
    df = remove_colum(df, 0.05)
    return df

def remove_colum(data_w, alpha):
    columns_to_remove = ['latitude', 'longitude', 'Unnamed: 0']
    data_w.drop(columns=[col for col in columns_to_remove if col in data_w.columns], inplace=True)

    x = data_w.drop(['price'], axis=1)
    y = data_w['price']
    x = pd.get_dummies(x, drop_first=True).astype(float)

    x, y = resample(x, y, n_samples=len(x) * 10, random_state=42)

    model = sm.OLS(y, x).fit()

    alpha = alpha
    p_values = model.pvalues
    insignificant_features = p_values[p_values > alpha].index.tolist()
    x_1 = x.drop(columns=insignificant_features)

    df_8 = pd.concat([y, x_1], axis=1)
    df_8.to_csv('data_2.csv')
    return df_8

def data_train(data_w):
    X = data_w.drop('price', axis=1)
    y = data_w['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

