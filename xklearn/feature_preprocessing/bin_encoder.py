import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .numeric_encoder import ContBinEncoder
from .categorical_encoder import CateBinEncoder



class BinEncoder(BaseEstimator, TransformerMixin):
    """
    将连续型变量转化为离散型（分箱）
    仅适用于cont， 支持缺失值
    """
    def __init__(self, diff_thr=20, num_of_bins=10, method='dt', cate_f=True, inplace=True, suffix='_bin', **kwargs):
        self.diff_thr = diff_thr
        self.num_of_bins = num_of_bins
        self.method = method
        self.inplace = inplace
        self.kwargs = kwargs
        self.cate_f = cate_f
        self.suffix = suffix
        self.kmap = {}


    def fit(self, X, y=None):
        X2 = X.apply(pd.to_numeric, errors='ignore')
        self.columns = X2.columns.tolist()
        self.categorical_cols = X2.select_dtypes('object').columns.tolist()
        self.numeric_cols = X2.select_dtypes(exclude='object').columns.tolist()

        # 数值型特征分箱
        if len(self.numeric_cols) > 0:
            self.numeric_bin_encoder = ContBinEncoder(diff_thr=self.diff_thr, num_of_bins=self.num_of_bins,
                                                      method=self.method,
                                                      inplace=self.inplace, suffix=self.suffix)
            self.numeric_bin_encoder.fit(X2[self.numeric_cols], y)
            self.kmap.update(self.numeric_bin_encoder)

        # 类别型特征分箱
        if len(self.categorical_cols) > 0 and self.cate_f:
            self.categorical_bin_encoder = CateBinEncoder(diff_thr=self.diff_thr, inplace=self.inplace, suffix=self.suffix)
            self.categorical_bin_encoder.fit(X2[self.categorical_cols], y)
            self.kmap.update(self.categorical_bin_encoder)

        return self


    def transform(self, X):
        check_is_fitted(self, attributes=['categorical_cols', 'numeric_cols', 'columns'])
        X2 = X.apply(pd.to_numeric, errors='ignore')
        if self.columns != X2.columns.tolist():
            raise ValueError(f'The columns should match the training data, but got {X2.columns.tolist()}')

        # 数值型特征分箱
        if len(self.numeric_cols) > 0:
            pass

        # 类别型特征分箱
        if len(self.categorical_cols) > 0 and self.cate_f:
            pass

        return




