import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.losses import MSE, binary_crossentropy

from sklearn import utils
from sklearn.utils.validation import (check_is_fitted, FLOAT_DTYPES, column_or_1d)

from functools import partial

from tensorflow.python.framework.errors_impl import (InvalidArgumentError as TensFlowInvalidArgumentError)

from ...utils.tensor_tools import to_tf_dataset, to_tf_tensor


def l1_norm(W, V, lambda_=0.001):
    """
    :param W: 线性部分参数
    :param V: 交叉部分参数
    :param lambda_:
    :return:
    """
    return tf.reduce_sum(tf.add(tf.multiply(lambda_, tf.abs(V)), tf.multiply(lambda_, tf.abs(W))))

def l2_norm(W, V, lambda_=0.001):
    """
    :param W: 线性部分参数
    :param V: 交叉部分参数
    :param lambda_:
    :return:
    """
    return tf.reduce_sum(tf.add(tf.multiply(lambda_, tf.pow(V, 2)), tf.multiply(lambda_, tf.pow(W, 2))))

def noop_norm(V, W, lambda_=0):
    return 0


def fm(X, w0, W, V):
    """
    模型预测函数
    :param X:  样本特征
    :param w0: bias权重
    :param W:  线性部分权重
    :param V:  交叉部分权重
    :return:
    """
    # 线性部分
    linear_terms = X * W  
    # 交叉部分
    interactions = tf.subtract(tf.pow(tf.tensordot(X, tf.transpose(V), 1), 2),
                               tf.tensordot(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)), 1)
                              )
    
    # 特征维度大于1
    if X.ndim > 1:
        linear_terms = tf.reduce_sum(linear_terms, 1, keepdims=True)
        interactions = tf.reduce_sum(interactions, 1, keepdims=True)
        
    # 只有一个特征
    else:
        linear_terms = tf.reduce_sum(linear_terms)
        interactions = tf.reduce_sum(interactions)
    
    return w0 + linear_terms + 0.5 * interactions


def train(train_dataset, num_factors=2, max_iter=1, 
          penalty=None, C=1.0, loss=None, optimizer=None, 
          random_state=42, dtype=tf.float32):
    """
    定义模型图
    :param train_dataset: 训练数据
    :param num_factors: 压缩后的维度，即k
    :param max_iter: 最大迭代次数
    :param penalty: 正则项 L1、L2等
    :param C: 正则参数
    :param loss: 损失函数
    :param optimizer: 优化器
    :param random_state: 随机种子
    :param dtype: 相关权重初始化的类型
    :return:
    """
    
    try:
        tf.random.set_seed(random_state)
    except:
        tf.set_random_seed(random_state)
    
    if C < 0:
        raise ValueError(f'Inverse regularization term must be positive; got (C={C})')
    if max_iter < 1:
        raise ValueError(f'max_iter must be > 0; got (max_iter={max_iter})')
    if num_factors < 1:
        raise ValueError(f'num_factors must be >= 1; got (num_factors={num_factors})')
        
    # 获取样本的特征维度
    p = train_dataset.element_spec[0].shape[1]
    
    
    # bias和线性部分权重初始化   (不需要初始化一些值么？)
    w0 = tf.Variable(tf.zeros([1], dtype=dtype))
    W = tf.Variable(tf.zeros([p], dtype=dtype))
    
    # 交叉部分权重
    V = tf.Variable(tf.random.normal([num_factors, p], mean=0, stddev=0.01, dtype=dtype, seed=random_state))
    
    
    for epoch_count in range(max_iter):
        for batch, (x, y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                pred = fm(x, w0, W, V)
                loss_ = loss(y, pred) + penalty(V, W, lambda_=1.0 / C)
            # 求梯度
            grads = tape.gradient(loss_, [w0, W, V])
            optimizer.apply_gradients(zip(grads, [w0, W, V]))
            # logger.debug(f'Epoch : {epoch_count}, batch : {batch}, loss : {loss_.numpy()}')
            los = round(loss_.numpy().mean(), 6)
            print(f'Epoch: {epoch_count}, batch: {batch} loss:, {los}')
    return w0, W, V
            


def to_tf_dataset(X, y, dtype=tf.float32, batch_size=256, shuffle_buffer_size=None):
    X = np.array(X)
    y = np.array(y)
    
    if X.shape[0] != y.shape[0]:
        raise ValueError('The number of training examples is not the same as the number of labels')
        
    dataset = tf.data.Dataset.from_tensor_slices((tf.cast(X, dtype=dtype), tf.cast(y, dtype=dtype))).batch(batch_size)
    
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    
    return dataset
    
    
def to_tf_tensor(X, dtype=tf.float32):
    X = np.array(X)
    return tf.cast(X, dtype=dtype)


class BaseFactorizationMachine(BaseEstimator):
    def __init__(self, num_factors=2, max_iter=32, eta=0.001, 
                 penalty='l2', C=1.0, batch_size=256,
                 random_state=42):
        self.num_factors = num_factors
        self.max_iter = max_iter
        if penalty and penalty not in ('l1', 'l2'):
            raise ValueError(f'penalty must be l1, l2 or None')
        self.penalty = penalty
        self.penalty_function = noop_norm

        # TODO: 增加正则系数
        if penalty:
            self.penalty_function = l2_norm if penalty == 'l2' else l1_norm

        self.eta = eta
        self.C = C
        self.batch_size = batch_size
        self.random_state = random_state


class FactorizationMachineRegressor(BaseFactorizationMachine, RegressorMixin):
    def __init__(self, num_factor=2, max_iter=32, eta=0.001, 
                 penalty='l2', C=1.0, batch_size=256, optimizer=None,
                 random_state=42):
        super().__init__(num_factor=num_factor,
                         max_iter=max_iter,
                         eta=eta,
                         penalty=penalty,
                         C=C,
                         batch_size=batch_size,
                         random_state=random_state)
        self.loss = MSE
        self.optimizer = optimizer
        
    def fit(self, X, y):
        """
        :param X:
        :param y:
        """
        X, y = utils.check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        column_or_1d(y)
        
        train_dataset = to_tf_dataset(X, y, batch_size=self.batch_size)
        if not self.optimizer:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.eta)
        self.w0_, self.W_, self.V_ = train(train_dataset,
                                           num_factors=self.num_factors,
                                           max_iter=self.max_iter,
                                           optimizer=self.optimizer,
                                           loss=self.loss,
                                           C=self.C,
                                           penalty=self.penalty_function,
                                           random_state=self.random_state)
        
        return self
    
    def predict(self, X):
        """
        :param X:
        :return y:
        """
        check_is_fitted(self, attributes=['w0_', 'W_', 'V_'])
        X = utils.check_array(X)
        X = to_tf_tensor(X)
        pred = fm(X, self.w0_, self.W_, self.V_).numpy()
        pred = column_or_1d(pred, warn=True)
        return pred
    
    def _more_tags(self):
        tags = super()._more_tags()
        tags['poor_score'] = True
        return tags



class FactorizationMachineClassifier(BaseFactorizationMachine, ClassifierMixin):
    def __init__(self, num_factors=2, max_iter=32, eta=0.001, 
                 penalty='l2', C=1.0, batch_size=256, optimizer=None,
                 random_state=42):
        super().__init__(num_factors=num_factors,
                         max_iter=max_iter,
                         eta=eta,
                         penalty=penalty,
                         C=C,
                         batch_size=batch_size,
                         random_state=random_state)
        self.loss = partial(binary_crossentropy, from_logits=True)
        self.optimizer = optimizer
        
    def fit(self, X, y):
        """
        :param X:
        :param y:
        """
        X, y = utils.check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        column_or_1d(y)
        
        self.label_binarizer = LabelBinarizer().fit(y)
        y = self.label_binarizer.transform(y)
        train_dataset = to_tf_dataset(X, y, batch_size=self.batch_size)
        self.classes_ = self.label_binarizer.classes_
        if not self.optimizer:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.eta,
                                                      beta_1=0.9,
                                                      beta_2=0.999,
                                                      epsilon=1e-07)
        
        self.w0_, self.W_, self.V_ = train(train_dataset,
                                           num_factors=self.num_factors,
                                           max_iter=self.max_iter,
                                           optimizer=self.optimizer,
                                           loss=self.loss,
                                           penalty=self.penalty_function,
                                           random_state=self.random_state)
        
        return self
    
    def predict_prob(self, X):
        """
        输出为sigmoid的概率数据
        :param X: 预测数据
        """
        check_is_fitted(self, attributes=['w0_', 'W_', 'V_'])
        X = utils.check_array(X)
        X = to_tf_tensor(X)
        try:
            pred = tf.nn.sigmoid(fm(X, self.w0_, self.W_, self.V_))
        except TensoFlowInvalidArgumentError as e:
            raise ValueError(str(e))
        return pred
    
    def predict(self, X, threshold=0.5):
        """
        输出是以threshold为准的0/1预测
        :param X: 预测数据
        :param thresold :预测的阈值
        """
        pred = self.predict_prob(X).numpy() > threshold
        return self.label_binarizer.inverse_transform(pred)
    
    def _more_tags(self):
        tags = super()._more_tags()
        tags['poor_score'] = True
        return tags