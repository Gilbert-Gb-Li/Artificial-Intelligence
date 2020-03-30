# $example on$
from pyspark.ml.classification import LogisticRegression, LinearSVC
# $example off$
from pyspark.sql import SparkSession
from configs import d_path

"""
An example demonstrating Logistic Regression Summary.
Run with:
  bin/spark-submit examples/src/main/python/ml/logistic_regression_summary_example.py
"""

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("LogisticRegressionSummary") \
        .getOrCreate()

    # Load training data
    training = spark.read.format("libsvm").load(f"{d_path}ml/mllib/sample_libsvm_data.txt")

    # training.printSchema()
    # root
    # | -- label: double(nullable=true)
    # | -- features: vector(nullable=true), 训练预测feature字段是必须的

    train, test = training.randomSplit([0.9, 0.1], seed=12345)

    svm = LinearSVC(maxIter=10, regParam=0.3, threshold=0.0)
    # Fit the model
    lsvcModel = svm.fit(train)

    print("Coefficients: " + str(lsvcModel.coefficients))
    print("Intercept: " + str(lsvcModel.intercept))

    # |features |       rawPrediction|   prediction|
    # |特 征    |         概率        |     预测值   |
    lsvcModel.transform(test).show()


    # $example off$

    spark.stop()
