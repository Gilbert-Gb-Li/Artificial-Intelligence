
from __future__ import print_function

from pyspark.sql import SparkSession
# $example on$
from pyspark.ml.feature import Binarizer
# $example off$

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("BinarizerExample")\
        .getOrCreate()

    # $example on$
    continuousDataFrame = spark.createDataFrame([
        (0, 0.1),
        (1, 0.8),
        (2, 0.2)
    ], ["id", "feature"])

    binarizer = Binarizer(threshold=0.5, inputCol="feature", outputCol="binarized_feature")

    binarizedDataFrame = binarizer.transform(continuousDataFrame)

    print("Binarizer output with Threshold = %f" % binarizer.getThreshold())
    binarizedDataFrame.show()
    # $example off$
    spark.stop()
