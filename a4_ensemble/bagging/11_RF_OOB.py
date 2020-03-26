import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
"""
obb error其实就是，随机森林里面某棵树，
    用来构建它的时候可能有n个数据没有用到，然后我们用这n个数据测一遍这棵树，然后obb error = 被分类错误数 / 总数
1）对每个样本，计算它作为oob样本的树对它的分类情况（约1/3的树）；
2）然后以简单多数投票作为该样本的分类结果；
3）最后用误分个数占样本总数的比率作为随机森林的oob误分率。

"""
print(__doc__)

RANDOM_STATE = 123
"""
# 生成数据
X, y = make_classification(n_samples=10000,        # 样本个数
                           n_features=25,          # 特征个数
                           n_informative=3,        # 有效特征个数
                           n_redundant=2,          # 冗余特征个数（有效特征的随机组合）
                           n_repeated=0,           # 重复特征个数（有效特征和冗余特征的随机组合）
                           n_classes=3,            # 样本类别
                           n_clusters_per_class=1, # 簇的个数
                           random_state=0)
"""
X, y = make_classification(n_samples=500, n_features=25, random_state=RANDOM_STATE)
# print(type(X), X.shape, y.shape)
# print(X[:1, :])
# print(y[:3])

ensemble_clfs = [
    ("RandomForestClassifier depth 10",
        RandomForestClassifier(max_depth=10, oob_score=True)),
    ("RandomForestClassifier depth 3",
        RandomForestClassifier(max_depth=3, oob_score=True))
]

''' OrderedDict, 带有顺序的字典 '''
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

min_estimators = 1
max_estimators = 2

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        '''添加参数'''
        clf.set_params(n_estimators=i)
        clf.fit(X, y)
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    print(label, xs, ys)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()