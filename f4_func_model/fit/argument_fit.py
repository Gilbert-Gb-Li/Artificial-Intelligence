from sklearn import datasets, neighbors, linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
n_samples = len(X_digits)
# X_digits.shape, y_digits.shape: (1797, 64) (1797,)

x_train = []
y_train = []
x_test = []
y_test = []

'''
参数： https://zhuanlan.zhihu.com/p/60983320
    loss：字符串，损失函数的类型。默认值为’hinge’
        ‘hinge’：合页损失函数，表示线性SVM模型
        ‘log’：对数损失函数，表示逻辑回归模型
        ‘modified_huber’：’hing’和’log’损失函数的结合，表现两者的优点
        ‘squared_hinge’：平方合页损失函数，表示线性SVM模型
        ‘perceptron’：感知机损失函数
    penalty：字符串，罚项类型
        ‘l2’：2-范数罚项，默认值，线性SVM的标准正则化函数
        ‘l1’：1-范数罚项
        ‘elasticnet’：l2和l1的组合。
    alpha：浮点数，罚项前的系数，默认值为0.0001。当参数learning_rate被设置成optimal的时候，该参数参与learning_rate值的计算
    l1_ratio：浮点数，elasticnet罚项中l2和l1的权重。取值范围0<=l1_ratio<=1。默认值为0.15
    fit_intercept：布尔值，是否估计截距，如果为假，认为数据已经中心化
    max_iter：整数，可选的。迭代的最大次数，只影响fit方法，默认值为5。从0.21版以后，如果参数tol不是空，则默认值为1000
    tol：浮点数或None，可选的。训练结束的误差边界。如果不是None，则当previous_loss-cur_loss<tol时，训练结束。默认值为None，从0.21版以后，默认值为0.001
    shuffle：布尔值，可选的。每轮迭代后是否打乱数据的顺序，默认为True
    verbose：整数，可选的，控制调试信息的详尽程度
    learning_rate：字符串，可选的。学习速率的策略
        ‘constant’：eta=eta0
        ‘optimal’：eta=1.0/(alpha*(t+t0))，默认值
        ‘invscaling’：eta=eta0/pow(t, power_t)
    eta0：浮点数，参与learning_rate计算，默认值为0
    power_t：参与learning_rate计算，默认值为0.5
    class_weight：词典{class_label:weight}或’balanced’或None，可选的。类别的权重。
        如果为None，则所有类的权重为1，’balanced’则根据y自动调节权重，使其反比于类别频率n_samples/(n_classes*np.bincount(y))
    warm_start：布尔值，可选的。设置为True时，使用之前的拟合得到的解继续拟合
    average：布尔值或整数，可选的。True时，计算平均SGD权重并存储于coef_属性中。
        设置为大于1的整数时，拟合使用过的样本数达到average时，开始计算平均权重
    n_jobs：整数，可选的。训练多元分类模型时，使用CPUs的数量，-1为使用全部，默认值为1
    random_state：打乱数据顺序的方式
attr：
    coef_：数组，shape=(1, n_features)二元分类；(n_classes, n_features)多元分类
    intercept_：数组，决策函数中常量b。shape=(1, )二元分类；(n_classes, )多元分类
    n_iter：整数，训练结束时，实际的迭代次数。对于多元分类来说，该值为所有二元拟合过程中迭代次数最大的
    loss_function_：使用的损失函数
function：
    decision_function(X)：对样本预测置信度得分
    densify()：将协方差矩阵转成数组
    fit(X, y[, coef_init, intercept_init_,…])：随机梯度下降法拟合线性模型
    get_params([deep])：返回分类器参数
    partial_fit(X, y[, classes, sample_weight])：增量拟合
    score(X, y[, sample_weight])：返回模型平均准确率
    set_params(*args, **kwargs)：设置模型参数
    sparsify()：将未知数矩阵w转成稀疏格式
'''
model = linear_model.SGDClassifier(loss='log')

kfold = KFold(n_splits=10, random_state=101, shuffle=True)
for x_train_index, x_test_index in kfold.split(X_digits):
    x_train.append(X_digits[x_train_index])
    x_test.append(X_digits[x_test_index])

for y_train_index, y_test_index in kfold.split(y_digits):
    y_train.append(y_digits[y_train_index])
    y_test.append(y_digits[y_test_index])

for i in range(len(x_train)):
    '''第一次调用需要添加classes类别'''
    model.partial_fit(x_train[i], y_train[i], classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

for i in range(len(x_test)):
    y_pred = model.predict(x_test[i])
    score = accuracy_score(y_test[i], y_pred)
    print('LogisticRegression score: %f', score)


# theta = logistic.coef_
# bias = logistic.intercept_
