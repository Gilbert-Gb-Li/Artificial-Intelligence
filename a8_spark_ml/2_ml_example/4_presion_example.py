from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib import evaluation
from pyspark.sql import SparkSession

if __name__ == "__main__":

    spark = SparkSession\
        .builder\
        .appName("MeasureExample")\
        .getOrCreate()

    # 我们先构造一个与上文一样的数据

    #          实际为真  | 实际为假
    # 预测为真      1    |   0
    # 预测为假      10   |   89
    # 左边是预测为真的概率，右边是真实值

    TP = [1.0, 1.0]
    TN = []
    for i in range(89):
        TN.append([0.0, 0.0])
    # print(len(TN))
    FP = []
    for i in range(89):
        TN.append([0.0, 1.0])

    all = TP + TN + FP


    # precision与recall得分计算，在pyspark版本未找到相应实现
    # 仅有ap与auc值的计算

    # 打印观察数据
    # scoreAndLabels = spark.parallelize(all)
    #  scoreAndLabels.collect().foreach(println)
    #  println(scoreAndLabels.count())
    #  到这里，我们构造了一个与上文例子一样的数据

    metrics = BinaryClassificationMetrics(all)
    # 下面计算的值，我们先只看右边的数，它表示计算的precision,recall,F1等
    # 左边是Threshold，后面会细说
    # (1.0, 1.0) precision跟我们自己之前计算的一样
    # (0.0, 0.11) 这是什么, 先不管
    # metrics.precisionByThreshold().collect()

    # print("---")

    # (1.0, 0.09090909090909091) recall跟我们自己之前计算的一样
    # (0.0,1.0)  先忽略
    # metrics.recallByThreshold().collect()
    # print("---")

    # (1.0, 0.16666666666666669) f1跟我们自己之前计算的一样
    # (0.0, 0.19819819819819817) 先忽略
    # metrics.fMeasureByThreshold().collect()
