import numpy as np
import pandas as pd
import func_model.visual.visuals as vs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# 载入北京房屋的数据集
from func_model.accuracy.accuracy import performance_metric
from func_model.param_search.grid_search import fit_model

data = pd.read_csv('../../titanic/boston/bj_housing2.csv')
prices = data['Value']
features = data.drop('Value', axis=1)
print("Beijing housing dataset has {} titanic points with {} variables each.".format(*data.shape))

# 观察原始数据，决定是否进行预处理
value = data['Value']
for item in ['Area', 'Room', 'Living', 'Year', 'School', 'Floor']:
    idata = data[item]
    plt.scatter(idata, value)
    # plt.show()

# 计算价值的最小值
minimum_price = np.min(prices)
# 计算价值的最大值
maximum_price = np.max(prices)
# 计算价值的平均值
mean_price = np.mean(prices)
# 计算价值的中值
median_price = np.median(prices)
# 计算价值的标准差
std_price = np.std(prices)
# 输出计算的结果
print("Statistics for Beijing housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))

# Train test split
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.20, random_state=0)
print("Train test split success!")

# 根据不同的训练集大小，和最大深度，生成学习曲线
vs.ModelLearning(X_train, y_train)
# 根据不同的最大深度参数，生成复杂度曲线
vs.ModelComplexity(X_train, y_train)

# 基于训练数据，获得最优模型
optimal_reg = fit_model(X_train, y_train)
# 输出最优模型的 'max_depth' 参数
print("Parameter 'max_depth' is {} for the optimal model.".format(optimal_reg.get_params()['max_depth']))

# 预测
predicted_price = optimal_reg.predict(X_test)
# 计算R^2的值
r2 = performance_metric(predicted_price, y_test)
print("Optimal model has R^2 score {:,.2f} on test titanic".format(r2))