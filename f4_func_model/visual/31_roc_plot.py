"""
样本在不同类别上的不均衡分布问题评估
# https://www.zhihu.com/question/39840928

?????
TP_rate = TP / (TP + FN) => 正样本预测正确部分 / 全部正样本
FP_rate = FP / (FP + TN) => 正样本预测错误部分 / 全部负样本
-------------------------------------------------------
TP_rate = TP / (TP + FN) => 正样本预测正确部分 / 全部正样本
TN_rate = TN / (FP + TN) => 负样本预测正确部分 / 全部负样本
roc_auc_score = (TP_rate + TN_rate) / 2
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc

y_test = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
y_pred1 = np.array([0.1, 0.2, 0.25, 0.7, 0.3, 0.9, 0.8, 0.7])
y_pred2 = np.array([0, 0, 1, 1, 1, 1, 2, 2, 0])


# 预测值是概率
# auc_score1 = roc_auc_score(y_test, y_pred1)
# print(auc_score1)

# 预测值是类别
# auc_score2 = roc_auc_score(y_test, y_pred2, pos_label=2)
# print(auc_score2)

# 多分类计算，需指定正样本标签
fpr, tpr, thresholds = roc_curve(y_test, y_pred2, pos_label=2)
print(fpr, tpr, thresholds)
auc_score3 = auc(fpr, tpr)
print(auc_score3)

# y_true = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
# y_score = [0.31689620142873609, 0.32367439192936548, 0.42600526758001989, 0.38769987193780364, 0.3667541015524296, 0.39760831479768338, 0.42017521636505745, 0.41936155918127238, 0.33803961944475219, 0.33998332945141224]
# thresholds = [0.42600526758001989, 0.42017521636505745, 0.41936155918127238, 0.39760831479768338, 0.38769987193780364, 0.3667541015524296, 0.33998332945141224, 0.33803961944475219, 0.32367439192936548, 0.31689620142873609]
#
# for threshold in thresholds:
#     # 大于等于阈值为1, 否则为0
#     y_prob = [1 if i>=threshold else 0 for i in y_score]
#     # # 结果是否正确
#     result = [i==j for i,j in zip(y_true, y_prob)]
#     # # 是否预测为正类
#     positive = [i==1 for i in y_prob]
#     #
#     tp = [i and j for i,j in zip(result, positive)] # 预测为正类且预测正确
#     fp = [(not i) and j for i,j in zip(result, positive)] # 预测为正类且预测错误
#
#     print(tp.count(True), fp.count(True))
#     print('result: ', result)
#     print('positive:', positive)

