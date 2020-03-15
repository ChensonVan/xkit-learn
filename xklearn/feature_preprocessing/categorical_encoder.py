import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.multiclass import type_of_target
from scipy import stats
from collections import defaultdict, Counter

class CountEncoder(BaseEstimator, TransformerMixin):
    """
    计数编码
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        pass


class CateLabelEncoder(BaseEstimator, TransformerMixin):
    """
    计数编码
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        pass


class CateOHEncoder(BaseEstimator, TransformerMixin):
    """
    类别型特征OneHot编码
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        pass

