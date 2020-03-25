print(__doc__)

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
print(X[:5])
y = np.sin(X)
print(y[:5, :])

X_train = X[0: int(len(X) * 0.9)]
Y_train = y[0: int(len(X) * 0.9)]

X_test = X[int(len(X) * 0.9):]
Y_test = y[int(len(X) * 0.9):]

# Fit regression model

'''(1) DT'''
regr_1 = DecisionTreeRegressor(max_depth=100)
regr_2 = DecisionTreeRegressor(max_depth=1)

'''(2) Linear regression'''
from sklearn import linear_model
regr_3 = linear_model.LinearRegression()

'''(3) SVM regression'''
from sklearn import svm
regr_5 = svm.SVR()

'''(4) KNN regression'''
from sklearn import neighbors
regr_6 = neighbors.KNeighborsRegressor()

'''(5) RF'''
from sklearn import ensemble
# 这里使用20个决策树
regr_7 = ensemble.RandomForestRegressor(n_estimators=20)

'''(6) GBDT'''
from sklearn import ensemble
regr_8 = ensemble.GradientBoostingRegressor(n_estimators=2000)

reg1Label = "reg1Label"
reg2Label = "reg2Label"
regr_1.fit(X, y)
regr_2.fit(X, y)


# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
# plt.scatter(X, y, s=20, edgecolor="black",
#             c="darkorange", label="data")
# plt.scatter(X_test, Y_test, s=20, edgecolor="black",
#             c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label=reg1Label, linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label=reg2Label, linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()