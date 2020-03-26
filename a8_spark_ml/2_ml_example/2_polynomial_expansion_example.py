from __future__ import print_function

# $example on$
from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml.linalg import Vectors
# $example off$
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("PolynomialExpansionExample")\
        .getOrCreate()

    # $example on$
    df = spark.createDataFrame([
        (Vectors.dense([2.0, 1.0]),),
        (Vectors.dense([0.0, 0.0]),),
        (Vectors.dense([3.0, -1.0]),)
    ], ["features"])

    polyExpansion = PolynomialExpansion(degree=2, inputCol="features", outputCol="polyFeatures")
    polyDF = polyExpansion.transform(df)

    polyDF.show(truncate=False)
    # $example off$

    spark.stop()
