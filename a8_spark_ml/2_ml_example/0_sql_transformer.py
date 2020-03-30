from __future__ import print_function

# $example on$
from pyspark.ml.feature import SQLTransformer
# $example off$
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("SQLTransformerExample")\
        .getOrCreate()

    # $example on$
    df = spark.createDataFrame([
        (0, 1.0, 3.0),
        (2, 2.0, 5.0)
    ], ["id", "v1", "v2"])
    # 创建新列
    sqlTrans = SQLTransformer(
        statement="SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")
    sqlTrans.transform(df).show()
    # $example off$

    spark.stop()
