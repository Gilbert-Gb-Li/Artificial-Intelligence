# from dataframe to dataframe
from __future__ import print_function
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from configs import d_path


if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("ctr_sample") \
        .getOrCreate()

    """1. Load training data"""
    df = spark\
        .read\
        .option("header", "true")\
        .option("inferschema", "true")\
        .option("mode", "DROPMALFORMED")\
        .format("com.databricks.spark.csv")\
        .load(f"{d_path}ctr/train_sample_ctr.csv").cache()

    """2. 特征工程"""
    df.printSchema()
    '''2.1 encoding stringindexer == pandas labelencoder'''
    df.select('site_domain').show(3)
    indexer = StringIndexer(inputCol="site_domain", outputCol="site_domainIndex")
    indexed = indexer.fit(df).transform(df)
    indexed.describe("site_domainIndex").show()

    """2.2 选择特征"""
    filtered = indexed.select("click", "site_domainIndex", "hour")
    filtered.describe().show()
    #
    # """2.3 将独立的特征列组合成向量"""
    assembler = VectorAssembler(inputCols=["site_domainIndex", "hour"], outputCol="features")
    output = assembler.transform(filtered)
    output = output.select("features", "click")
    output.show(n=3)

    """3. 切分训练测试集"""
    (trainingData, testData) = output.randomSplit([0.7, 0.3])

    """4. Train a DecisionTree model."""
    lr = LogisticRegression(labelCol="click", featuresCol="features",
                            maxIter=10, regParam=0.3, elasticNetParam=0.8)

    """5. Fit the model"""
    model = lr.fit(trainingData)

    """6. Make predictions."""
    predictions = model.transform(testData)

    """
    # 计算准确度: MulticlassClassificationEvaluator
    # 列名需指定为 "prediction", "label"
    """
    predictions = predictions.withColumnRenamed("click", "label")
    predictions.select("prediction", "label", "features").show(5)
    predictionAndLabels = predictions.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
    spark.stop()