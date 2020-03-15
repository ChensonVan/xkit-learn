import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _get_max_same_count(c):
    try:
        return c.value_counts().iloc[0]
    except:
        return len(c)


def _get_same_value_ratio(X):
    t = X.apply(_get_max_same_count) / X.shape[0]
    t.name = 'same_value'
    return t


def _get_missing_value_ratio(X):
    t = X.isnull().mean()
    t.name = 'missing'
    return t


class BaseEncoder(BaseEstimator, TransformerMixin):
    """
    用于剔除缺失值严重列，同值严重列，不同值严重cate列（字符串列如果取值太过于分散，则信息量过低）
    适用于cont和cate，支持缺失值, 建议放置在encoder序列第一位次
    """
    def __init__(self, missing_thr=0.8, same_thr=0.8, cate_thr=0.9):
        self.missing_thr = missing_thr
        self.same_thr = same_thr
        self.cate_thr = cate_thr


    def fit(self, X, y=None):
        X2 = X.apply(pd.to_numeric, errors='ignore')
        numeric_cols = X2.dtypes.map(is_numeric_dtype).index.tolist()
        categorial_cols = X2.columns.difference(numeric_cols).tolist()

        # 计算缺失值严重列
        tmp = _get_missing_value_ratio(X2)
        self.missing_cols = list(tmp[tmp > self.missing_thr].index.values)

        # 计算同值严重列
        tmp = _get_same_value_ratio(X2)
        self.same_cols = list(tmp[tmp > self.same_thr].index.values)

        # 计算不同值严重cate列
        if len(categorial_cols) > 0:
            tmp = X2[categorial_cols]
            tmp = tmp.nunique() / X2.shape[0]
            self.cate_cols = list(tmp[tmp > self.cate_thr].index.values)
        else:
            self.cate_cols = list([])

        self.drop_cols = list(set(self.missing_cols + self.same_cols + self.cate_cols))
        return self


    def transform(self, X):
        check_is_fitted(self, attributes=['missing_cols', 'same_cols', 'drop_cols'])
        return X.drop(self.drop_cols, axis=1)


    def fit_transform(X, y=None):
        pass



