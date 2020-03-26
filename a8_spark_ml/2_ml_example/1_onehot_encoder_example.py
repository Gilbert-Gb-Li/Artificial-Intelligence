
from __future__ import print_function

# $example on$
from pyspark.ml.feature import OneHotEncoder, StringIndexer
# $example off$
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("OneHotEncoderExample")\
        .getOrCreate()

    # $example on$
    df = spark.createDataFrame([
        (0, "a"),
        (1, "b"),
        (2, "c"),
        (3, "a"),
        (4, "a"),
        (5, "c"),
        (6, "b"),
        (7, "d")
    ], ["id", "category"])

    stringIndexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
    model = stringIndexer.fit(df)
    indexed = model.transform(df)

    encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
    encoded = encoder.transform(indexed)
    encoded.show()
    # $example off$

    spark.stop()
