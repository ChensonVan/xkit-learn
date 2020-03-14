import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.multiclass import type_of_target
from scipy import stats



def get_cut_points_by_tree(x, y, criterion='gini', max_depth=3, min_samples_leaf=0.01, 
                           max_leaf_nodes=None, random_state=42, precision=4):
    """
    根据决策树选出cut_points
    """

    # 二分类
    if type_of_target(y) == 'binary':
        clf = DecisionTreeClassifier(
                                    # criterion=criterion,
                                    max_depth=max_depth, 
                                    min_samples_leaf=min_samples_leaf, 
                                    max_leaf_nodes=max_leaf_nodes, 
                                    random_state=random_state)
    # 回归
    else:
        clf = DecisionTreeRegressor(
                                    # criterion=criterion,
                                    max_depth=max_depth, 
                                    min_samples_leaf=min_samples_leaf, 
                                    max_leaf_nodes=max_leaf_nodes, 
                                    random_state=random_state)

    clf.fit(np.array(x).reshape(-1, 1), np.array(y))
    th = clf.tree_.threshold.round(precision)
    fea = clf.tree_.feature
    # -2 代表的是
    return sorted(th[np.where(fea != -2)])


def get_cut_points_by_monotonic(x, y, num_of_bins=10, precision=4):
    """
    根据数据单调性选出切分点（不严谨，非卡方分箱，需重构）
    """
    x, y = pd.Series(x), pd.Series(y)
    x_notnull = x[pd.notnull(x)]
    y_notnull = y[pd.notnull(x)]
    r = 0
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"x": x_notnull, "y": y_notnull, 
                           "Bucket": pd.qcut(x_notnull, num_of_bins, duplicates='drop')})
        d2 = d1.groupby('Bucket', as_index=True)
        r, p = stats.spearmanr(d2['x'].mean(), d2['y'].mean())
        num_of_bins -= 1
    # d3 = pd.DataFrame(d2['x'].min(), columns=['min_x'])
    # print(d2['x'].min().tolist()[:1] + d2['x'].max().tolist())
    # return d2['x'].min().tolist()[:1] + d2['x'].max().tolist()
    # d3['max_x'] = d2['x'].max()
    # d3['min_x'] = d2['x'].min()
    # d3['sum_y'] = d2.sum().y
    # d3['count_y'] = d2.count().y
    # d3['mean_y'] = d2.mean().y
    # d4 = (d3.sort_index(by='min_x')).reset_index(drop=True)
    # return d4
    return d2['x'].min().tolist()[:1] + d2['x'].max().tolist()


def get_cut_points_by_freq(x, num_of_bins=10, precision=4):
    """
    根据数据等频率选择切分点，若无法根据指定的num_of_bins，则近视切分
    """
    x = pd.Series(x)
    x = x[pd.notnull(x)]
    interval = 100 / num_of_bins
    cp = sorted(set(np.percentile(x,  i * interval) for i in range(num_of_bins + 1)))
    cp = np.round(cp, precision).tolist()
    return sorted(set(cp))


def get_cut_points_by_interval(x, num_of_bins=10, precision=4):
    """
    等间距选取切分点
    """
    cp = pd.cut(x, num_of_bins, retbins=True)[1]
    return np.round(cp, precision).tolist()


def get_cut_points(x, y=None, num_of_bins=10, method='qcut', precision=4, random_state=42, **kwargs):
    if method == 'qcut':
        return get_cut_points_by_freq(x, num_of_bins=num_of_bins, precision=precision)
    elif method == 'cut':
        return get_cut_points_by_interval(x, num_of_bins=num_of_bins, precision=precision)
    elif method == 'dt':
        return get_cut_points_by_tree(x, y, random_state=random_state, precision=precision, **kwargs)
    elif method == 'mono':
        return get_cut_points_by_monotonic(x, y, num_of_bins=num_of_bins, precision=precision)
    else:
        raise ValueError(f'The method of bin should be in [dt, qcut, cut, mono], but got {self.method}') 


class ContBinEncoder(BaseEstimator, TransformerMixin):
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
