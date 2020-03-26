print(__doc__)

from sklearn import datasets, neighbors, linear_model
"""
手写数字识别
---------
1.导入数据
Load and return the digits dataset (classification).

    Each datapoint is a 8x8 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class             ~180
    Samples total                 1797
    Dimensionality                  64
    =================   ==============
"""
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
n_samples = len(X_digits)
# X_digits.shape, y_digits.shape: (1797, 64) (1797,)

"""
2.切分训练集
拿90%作训练集
"""
X_train = X_digits[:int(.9 * n_samples)]
y_train = y_digits[:int(.9 * n_samples)]
X_test = X_digits[int(.9 * n_samples):]
y_test = y_digits[int(.9 * n_samples):]
# print(X_test.shape)
# print(X_test[0])

"""
3.模型调用
参数：
    penalty：惩罚项，可为'l1' or 'l2'
    multi_class：多分类'ovr' or 'multinomial'。'multinomial'即为MvM。
    C：正则化系数。并非添加到系数矩阵上，所以其值越小正则化越强
    tol：优化算法停止的条件。当迭代前后的函数差值小于等于tol时就停止，默认为‘1e-4’
    class_weight：用于标示分类模型中各种类型的权重，
        {class_label: weight} 如,class_weight={0: 0.7, 1: 0.3}
        balanced: 自动计算n_samples / (n_classes * np.bincount(y))
            计算不同元素的个数, np.bincount([0,0,1,2,2])输出是[2 1 2]
    max_iter: 算法收敛最大迭代次数，int类型，默认为100
    -------------
    dual：选择目标函数为原始形式还是对偶形式。
    solver：逻辑回归损失函数的优化方法
"""

logistic = linear_model.LogisticRegression(C=10)
'''4.测试'''
print('LogisticRegression score: %f'
      % logistic.fit(X_train, y_train).score(X_test, y_test))

'''
coef_       : 逻辑回归权重W
intercept_  : 逻辑回归偏置b
'''
theta = logistic.coef_
bias = logistic.intercept_
