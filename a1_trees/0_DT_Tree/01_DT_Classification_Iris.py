"""
决策树
---------
不适合高维稀疏矩阵
"""

from sklearn.datasets import load_iris
from sklearn import tree
import os
import pydot

'''返回当前工作目录'''
print(os.getcwd())

"""
# scikit-learn决策树算法类库内部实现是使用了调优过的CART树算法，
# 既可以做分类，又可以做回归。分类决策树的类对应的是DecisionTreeClassifier，
# 而回归决策树的类对应的是DecisionTreeRegressor。两者的参数定义几乎完全相同，但是意义不全相同。

# As with other classifiers, DecisionTreeClassifier takes as input two arrays: an array X,
# of size [n_samples, n_features] holding the training samples,
# and an array Y of integer values, size [n_samples], holding the class labels for the training samples:

# 主要参数讲解
1. Criterion,分类纯度算法
2. random_state,随机种子
3. max_depth,限制树的最大深度，超过设定深度的树枝全部剪掉，实际⽤用时建议从3开始尝试
4. min_samples_leaf,每个子结点都必须包含至少样本数量，否则分支就不会发生
5. min_samples_split,要包含至少指定样本，这个结点才允许被分支
6. class_weight, 样本平衡参数
"""
clf = tree.DecisionTreeClassifier(criterion="entropy")
iris = load_iris()
print(iris.data[0:5])
print(iris.target[0:5])

clf = clf.fit(iris.data, iris.target)

"""
1. 将模型导出描述文件
2. 生成图
3. 导出图
"""
tree.export_graphviz(clf, out_file='./tree.dot')
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')