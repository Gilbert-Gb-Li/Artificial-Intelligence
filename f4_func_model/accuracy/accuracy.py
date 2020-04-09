import numpy as np
from sklearn.metrics import r2_score, accuracy_score, recall_score, precision_score, f1_score


def performance_metric(y_true, y_predict):
    """
    计算并返回预测值相比于预测值的分数
    https://blog.csdn.net/wade1203/article/details/98477034
    R² =（1-u/v）。
    u=((y_true - y_pred) ** 2).sum()
    v=((y_true - y_true.mean()) ** 2).sum()
    其中y_pred已经在score方法中通过predict()方法得到，再与y_true进行比对。
    所以y_true和y_pred越接近，u/v的值越小。
    R2的值就越大！
    """
    score = r2_score(y_true, y_predict, sample_weight=None, multioutput=None)
    return score


def accuracy_count(y_pred, y_test):
    # 方法一
    acc = len(np.where(y_pred == y_test)[0]) / len(y_test)
    # 方法二
    correct = np.sum(y_test == y_pred)
    n = len(y_test)
    acc = float(correct) / n
    # 方法三
    from sklearn.metrics import accuracy_score
    y_pred = [0, 2, 2]
    y_true = [0, 1, 2]
    acc = accuracy_score(y_true, y_pred)
    return acc




def pre_re_f1():
    y_test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    predictions = [0, 0, 1, 1, 0, 0, 0, 2, 2, 0, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2]
    y_true = [0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    """----------------
    分类问题常用评价指标
    不仅适合二分类，也适合多分类。只需要指出参数average=‘micro’/‘macro’/'weighted’
        # binary:  只对二分类问题有效，返回由pos_label指定的类的f1_score。
        # macro：先在各混淆矩阵上分别计算各类的查准率，查全率和F1，然后再计算平均值（各类别F1的权重相同）
            => 对每一类别的f1_score进行简单算术平均
        # micro： 通过先计算总体的TP，FN和FP的数量，再计算F1
        # 'weighted': 对每一类别的f1_score进行加权平均，权重为各类别数在y_true中所占比例。
    --------------------"""

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    F1 = f1_score(y_true, y_pred)
    print("accuracy:", accuracy, '\n', "precision:", precision, '\n', "recall:", recall, '\n', "F1 :", F1)


def mean_squared_error():
    # 2. 均方误差
    from sklearn.metrics import mean_squared_error
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    print(mean_squared_error(y_true, y_pred))
    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]
    print(mean_squared_error(y_true, y_pred))



if __name__ == '__main__':
    pre_re_f1()
