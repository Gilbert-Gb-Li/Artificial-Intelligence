import numpy as np

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

digits = load_digits()
X, y = digits.data, digits.target
clf = RandomForestClassifier(n_estimators=20)
"""======================
参数搜索
--------
# RandomSearchCV 随机搜索
# GridSearchCV   网格搜索
# 参数：
    estimator, 分类器
    param_grid | param_distributions, 参数列表
    cv=5, 交叉验证参数，默认None，使用3折交叉验证
    n_iter, RandomizedSearchCV迭代步骤
    scoring, 评分方法，默认为None，即estimator的默认评分标准， 分类器accuracy_score， 回归器r2_score
        sklearn.metrics提供的评分标准：
        分类
        accuracy	        metrics.accuracy_score
        average_precision	metrics.average_precision_score
        f1	                metrics.f1_score
        f1_micro	        metrics.f1_score
        f1_macro	        metrics.f1_score
        f1_weighted     	metrics.f1_score
        f1_sample	        metrics.f1_score
        neg_log_loss	    metrics.log_loss
        precision	        metrics.precision_score
        recall	            metrics.recall_score
        roc_auc	            metrics.roc_auc_score
        聚类
        adjusted_rand_score	        metrics.adjusted_rand_score
        回归
        neg_mean_absolute_erroe	    metrics.neg_mean_absolute_erroe
        neg_mean_squared_error	    metrics.neg_mean_squared_error
        neg_median_absolute_error	metrics.neg_median_absolute_error
        r2	                        metrics.r2
        自定义评分
        scoring_fnc = make_scorer(performance_metric)
            def performance_metric():
                score = r2_score(y_true, y_predict, sample_weight=None, multioutput=None)
                return score
# 属性：
    cv_results_     : 属性列表，每次训练的相关信息 
    best_estimator_ : estimator或dict；效果最好的分类器
    best_params_    : dict,在保持数据上给出最佳结果的参数设置
    best_score_     : best_estimator的平均交叉验证分数
    best_index_     : 对应于最佳候选参数设置的索引(cv_results_数组的索引)。
==========================="""


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        '''
        # np.flatnonzero, 返回指定元素（非零）的位置
        # d = np.array([1,2,3,4,5,3])
        # idx = np.flatnonzero(d == 3)
        # idx: [2, 5]
        '''
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


"""---------------------------------------------------------------
RandomSearchCV 随机搜索，针对连续性数值，其搜索策略如下：
（a）对于搜索范围是distribution[连续]的超参数，根据给定的distribution随机采样；
（b）对于搜索范围是list[离散]的超参数，在给定的list中等概率采样；
（c）对a、b两步中得到的n_iter组采样结果，进行遍历。
（d）如果给定的搜索范围均为list，则不放回抽样n_iter次。
--------------------------------------------------------------------"""
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5)
start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)


"""------------
网格搜索
---------------"""
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)


