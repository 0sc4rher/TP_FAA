from sklearn import datasets
from joblib import dump, load
from sklearn.metrics import accuracy_score
from flask import Flask
from flask_restful import reqparse, Api, Resource
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import pandas
import xgboost


def get_keys():
        clf = load('modelapi.joblib')
        def keywords_api(X):
                y_pred = str(clf.predict([X]))
                return y_pred
        return keywords_api
