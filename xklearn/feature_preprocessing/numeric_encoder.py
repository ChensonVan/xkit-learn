import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.multiclass import type_of_target
from scipy import stats
from .utils import get_cut_points


class ContBinEncoder(BaseEstimator, TransformerMixin):
    """
    针对连续型特征分箱，支持树模型、等频、等距、单调性分箱等
    """
    def __init__(self, diff_thr=20, num_of_bins=10, method='qcut', suffix='_bin', precision=4, **kwargs):
        self.diff_thr = diff_thr       # 特征列唯一值个数的限制，如果唯一值过于少，则忽略不分箱
        self.num_of_bins = num_of_bins
        self.method = method
        self.suffix = suffix           # 分箱后特征的后缀
        self.precision = precision
        self.kwargs = kwargs           # 用于dt的入参
        self.cut_point_map = {}        # 
        self.kmap = {}                 # 


    def fit(self, X, y=None):
        # bad case检查
        if y is None:
            if self.method in ['dt', 'mono']:
                raise ValueError(f'The target should be supply for method of dt and mono, but got None')

        # TODO: 检查X的特征类型是否为数值型
        X2 = X.apply(pd.to_numeric, errors='ignore')
        self.columns = X2.columns.tolist()
        self.categorical_cols = X2.select_dtypes('object').columns.tolist()
        self.numeric_cols = X2.select_dtypes(exclude='object').columns.tolist()
        if len(self.numeric_cols) == 0:
            raise ValueError(f'The num of numeric columns should > 0, but got zeros.')

        for col in self.numeric_cols:
            # TODO: 头尾如何处理
            self.cut_point_map[col] = get_cut_points(X2[col], y, 
                                                   num_of_bins=self.num_of_bins,
                                                   method=self.method, 
                                                   precision=self.precision)
        return self


    def transform(self, X):
        check_is_fitted(self, attributes=['categorical_cols', 'numeric_cols', 'columns'])
        X2 = X.apply(pd.to_numeric, errors='ignore')
        if self.columns != X2.columns.tolist():
            raise ValueError(f'The columns should match the training data, but got {X2.columns.tolist()}')

        result = {}
        for col in self.numeric_cols:
            # TODO: 需要检查一下cut_point的个数先
            cut_point = [-np.inf] + self.cut_point_map[col][1:-1] + [np.inf]
            result[col] = pd.cut(X2[col], bins=cut_point).astype(str).tolist()
        df_result = pd.DataFrame(result).add_suffix(self.suffix)
        return df_result


class ContBinWOEEncoder(BaseEstimator, TransformerMixin):
    """
    先对连续型特征分箱，后转WOE编码，将分箱后数据与标签线性对齐
    """
    pass


class ContBinOHEncoder(BaseEstimator, TransformerMixin):
    """
    先对连续型特征分箱，后转OneHot编码，将分箱后数据稀疏化
    """
    pass