import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

def estimator_report(model_type, data_type, y_train, y_pred):
    matrix = confusion_matrix(y_train, y_pred)
    TN = matrix[0][0]
    FP = matrix[0][1]
    FN = matrix[1][0]
    TP = matrix[1][1]
    accuracy = (TP+TN) / (TP+TN+FP+FN)
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    d = {'model_type': model_type, 'data_type': data_type, 
        'accuracy': accuracy, 'precision': precision, 'recall': recall}
    return d

# %%
def polyfeatures(X_train_scaled, y_train, X_validate_scaled, y_validate, n):
    pf = PolynomialFeatures(degree=n)
    pf = pf.fit(X_train_scaled)
    X_train_scaled = pf.transform(X_train_scaled)
    X_validate_scaled = pf.transform(X_validate_scaled)
    lr = LogisticRegression()
    lr = lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_train_scaled)
    d1 = estimator_report(n, 'train', y_train, lr_pred)
    lr_pred = lr.predict(X_validate_scaled)
    d2 = estimator_report(n, 'validate', y_validate, lr_pred)
    return d1, d2