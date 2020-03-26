import csv
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

dates = []
prices = []

"""取日期的天及开盘价"""
def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader) 	# skipping column names
		for row in csvFileReader:
			# 将日期的天及开盘价放入列表
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return

def predict_price(dates, prices, x):
	# 等价于：
	# dates = np.array(dates)
	# dates = dates[:, None]
	# dates = dates[:, np.newaxis]
	dates = np.reshape(dates, (len(dates), 1))	 # converting to matrix of n X 1
	prices = np.reshape(prices, (len(prices), 1))
	# dates.shape: (19, 1)
	# dates[0:2]
	# 	[[26]
	# 	 [25]]

	# GBDT算法
	from sklearn import ensemble
	# linear_mod = ensemble.GradientBoostingRegressor(n_estimators=100)
	linear_mod = linear_model.LinearRegression()

	linear_mod.fit(dates, prices)

	plt.scatter(dates, prices, color='black', label='Data')
	plt.plot(dates, linear_mod.predict(dates), color='red', label='Linear model')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title(' Regression')
	plt.legend()
	plt.show()

	# 预测的输入类型为【样本，特征】矩阵
	x = np.reshape(x, (1, 1))
	return linear_mod.predict(x)[0][0], linear_mod.coef_[0][0], linear_mod.intercept_[0]


""" 1. 数据读取 """
root = '../../data'
get_data(f"{root}/goog.csv")
print("Dates- ", dates)
# Dates-  [26, 25, 24, 23, 22, 19, 18, 17, 16, 12, 11, 10, 9, 8, 5, 4, 3, 2, 1]
print("Prices- ", prices[0: 2])
# Prices-  [708.58, 700.01]

predicted_price, coefficient, constant = predict_price(dates, prices, 29)
print("\nThe stock open price for 29th Feb is: $", str(predicted_price))
print("The regression coefficient is ", str(coefficient), ", and the constant is ", str(constant))
print("the relationship equation between dates and prices is: price = ", str(coefficient), "* date + ", str(constant))