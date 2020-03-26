from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
# names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
"""
boston 房价各种训练模型比较
"""
print(__doc__)

from sklearn.datasets import load_boston

boston = load_boston()

X = boston.data
Y = boston.target

""" 定义交叉验证 """
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)


""" 反射调用？？"""
def bulid_model(model_name):
    model = model_name()
    return model


scoring = 'mean_squared_error'

"""各模型比较"""
# name                          mean                time
# LinearRegression          0.20252899006055305   0.04607399999999995
# Ridge                     0.25616687037593344   0.0392809999999999
# Lasso                     0.19828974626177504   0.03580400000000017
# ElasticNet                0.22774942615238083   0.035636
# KNeighborsRegressor       -4.949260514859802    0.04061500000000007
# DecisionTreeRegressor     -0.09846441974802371  0.10510200000000003
# SVR                       -1.0050199190593039   0.2560889999999998

for model_name in [LinearRegression, Ridge, Lasso, ElasticNet, KNeighborsRegressor, DecisionTreeRegressor, SVR]:
    import time

    starttime = time.clock()
    model = bulid_model(model_name)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='r2')
    print(model_name)
    print(results.mean(), results.std() ** 2)
    # long running
    endtime = time.clock()
    print('Running time: %s Seconds' % (endtime - starttime))
