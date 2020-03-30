import tensorflow as tf
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
"""===========================================
KerasClassifier (KerasRegressor)类包装Keras模型
由此调用sklearn的方法，例网格搜索
=============================================="""


def create_model():
    '''
    生成keras 模型
    :return:
    '''
    return Sequential()

KerasClassifier(build_fn=create_model(), )