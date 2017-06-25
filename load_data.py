"load data, which should be a dictionary with"
"x_train, y_train, x_test, y_test - numpy arrays"
"defs files import data from here"

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

type_map = {'smoke': np.float, 'alco': np.float, 'active': np.float}
train = pd.read_csv('../data/raw/train.csv', sep=';')

def cat_col(data, col, bins):
    hist = np.histogram(data[col], bins=bins)
    data[col] = np.digitize(data[col], hist[1])
    return data[col]


def preprocess_data(data):
    result = data.copy()

    result.drop(result[(result['ap_hi'] < 0) | (result['ap_lo'] < 0)].index) # ap < 0 is clearly an error
    result.drop(result[(result['ap_hi'] >= 300) | (result['ap_lo'] >= 200)].index) # ap < 0 is clearly an error

    result = pd.concat([result, pd.get_dummies(data['cholesterol'], prefix='chol')], axis=1)
    result['age'] /= 365
    result['gender'] = result['gender'] == 1
    result.drop(['id', 'cardio', 'cholesterol'], inplace=True, axis=1)
    result['age'] = cat_col(result, 'age', 40)
    result['height'] = cat_col(result, 'height', 40)
    result['weight'] = cat_col(result, 'weight', 40)

    return result

x = preprocess_data(train)
y = train['cardio']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
