# -*- coding: utf-8 -*-
"""
数据预处理
如前面所说，我们的数据预处理工作占用了我们的70%时间
其完成质量直接影响最终结果
首先需要对数据有个整体的认识
"""
# 加载相关模块和库
import sys
import io
#改变标准输出的默认编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

import pandas as pd

# 读入后直接是dataframe
data_train = pd.read_csv("./data/train.csv")

# 列名信息
print("看列名", data_train.columns)
"""
# PassengerId => 乘客ID    
# Survived => 获救情况（1为获救，0为未获救）  
# Pclass => 乘客等级(1/2/3等舱位) 
# Name => 乘客姓名     
# Sex => 性别     
# Age => 年龄     
# SibSp => 堂兄弟/妹个数     
# Parch => 父母与小孩个数 
# Ticket => 船票信息     
# Fare => 票价     
# Cabin => 客舱     
# Embarked => 登船港口 
"""

# 数据摸底
# 问题1 每列的数量 --> 空值填充
# 问题2 每列类型 --> 类别型数据处理
print("看每列性质，空值和类型", data_train.info())

# 查看统计信息
# 问题1 每列的均值与标准差
    # 看下差距 --> 异常值处理与归一化
    # 均值分位数 --> 异常点
# 问题2 
print("看每列统计信息", data_train.describe())


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']     # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号

# plt.subplot2grid((2, 3), (1, 0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')    # sets our legend for our graph.
plt.show()
