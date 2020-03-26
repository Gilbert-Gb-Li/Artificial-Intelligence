## 笔记

### ALS
#### 介绍
- 原理，参见PDF
- 过程
    - 创建评分矩阵，ratings [userId, itemId, score, timestamp]
    - fit数据
    - transform并RegressionEvaluator计算得分
    - 推荐
        - recommendForAllUsers
        - recommendForAllItems
        - ...
#### 注意点
- 冷启动策略
    可以设置coldStartStrategy为drop来去除新用户的nan值

## 代码
0_test：测试环境安装实例
1_pyspark_dataframe：pyspark dataframe api 练习实例
2_spark_kdd19: pyspark入侵检测实战
3_pyspark-recommendation：spark推荐算法