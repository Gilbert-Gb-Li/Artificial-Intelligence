import pandas as pd
import numpy as np
from configs import d_path
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

"""
点击率预测
=========
- 训练集测试集切分
- 参数搜索
- one-hot
"""
def gridSearch(X_train, y_train):
    """参数搜索"""
    grid = {'C': np.logspace(-3, 3, 7), 'penalty': ["l1", "l2"]}
    logReg = linear_model.LogisticRegression()
    logRegCv = GridSearchCV(logReg, grid, cv=10)
    logRegCv.fit(X_train, y_train)

    print("tuned hpyerparameters :(best parameters) ", logRegCv.best_params_)
    print("accuracy :", logRegCv.best_score_)


def oneHotEncoding(data_train):
    print(data_train.info())
    # print(data_train[['site_domain', 'site_category', 'banner_pos']].describe())
    site_domain = pd.get_dummies(data_train['site_domain'], prefix='site_domain')
    site_category = pd.get_dummies(data_train['site_category'], prefix='site_category')
    df = pd.concat([data_train, site_domain, site_category], axis=1)
    df.drop(['site_domain', 'site_category'], axis=1, inplace=True)

    train_df = df.filter(regex='click|site_domain_.*|site_category_.*|hour|C1|banner_pos')
    return train_df.values

# 1. 数据读取
data_train = pd.read_csv(f"{d_path}ctr/train_sample_ctr.csv")


# 2.提取特征、标签
train_np = oneHotEncoding(data_train)
# y: click
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]

# 3. 数据集切分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 4. 参数搜索
gridSearch(X_train, y_train)

# 5. 模型选择及训练
clf = linear_model.LogisticRegression(C=0.01, penalty="l2")
clf.fit(X_train, y_train)

# 6. 预测及准确率
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))