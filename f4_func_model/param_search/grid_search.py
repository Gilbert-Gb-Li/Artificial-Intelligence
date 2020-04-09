import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from func_model.accuracy.accuracy import performance_metric


def knn_search(X_test, y_test):
    """
    KNN 测试最佳邻居数
    :param X_test:
    :param y_test:
    :return:
    """

    k_range = range(1, 31)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        y_pred = knn.predict(X_test)
        scores = knn.score(X_test, y_test)
        k_scores.append(scores.mean())

    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()


def fit_model(X, y):
    """
    # 基于输入数据 [X,y]，利于网格搜索找到最优的决策树模型
    # 自动调参，适合小数据集。相当于写一堆循环，自己设定参数列表，一个一个试，找到最合适的参数。
    # 参数：
        # estimator：所使用的分类器
        # param_grid：值为字典或列表，即需要最优化的参数的取值
        # scoring：准确评价标准
        # cv：交叉验证参数
        # 属性方法：
        # grid.fit(train_x, train_y)：运行网格搜索
        # grid_scores_：给出不同参数情况下的评价结果
        # best_params_：描述了已取得最佳结果的参数的组合
        # best_score_：成员提供优化过程期间观察到的最好的评分
    """
    regressor = DecisionTreeRegressor()

    cross_validator = KFold(n_splits=10, shuffle=False, random_state=None)

    params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

    '''自定义评价标准'''
    scoring_fnc = make_scorer(performance_metric)

    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cross_validator, verbose=1)
    # 基于输入数据 [X,y]，进行网格搜索
    grid = grid.fit(X, y)
    ''' ##
    返回网格搜索后的最优模型 
    ## '''
    return grid.best_estimator_

def gridBestModel(X, y):
    """
    # grid.best_estimator_: 返回最佳模型， 可以直接用于预测
    # .get_params(): 返回最佳参数
    """
    optimal_reg = fit_model(X, y)
    # 输出最优模型的 'max_depth' 参数
    print("Parameter 'max_depth' is {} for the optimal model.".format(optimal_reg.get_params()['max_depth']))
