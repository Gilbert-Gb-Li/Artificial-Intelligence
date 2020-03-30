"""
An example of how to use DataFrame for ML. Run with::
    bin/spark-submit examples/src/main/python/ml/dataframe_example.py <input>
"""
from __future__ import print_function
import os
import sys
import tempfile
import shutil
from pyspark.sql import SparkSession
from pyspark.mllib.stat import Statistics
from pyspark.mllib.util import MLUtils

if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage: dataframe_example.py <libsvm file>", file=sys.stderr)
        exit(-1)
    elif len(sys.argv) == 2:
        input = sys.argv[1]
    else:
        input = "data/mllib/sample_libsvm_data.txt"

    spark = SparkSession \
        .builder \
        .appName("DataFrameExample") \
        .getOrCreate()

    # Load input data
    print("Loading LIBSVM file with UDT from " + input + ".")
    df = spark.read.format("libsvm").load(input).cache()
    print("Schema from LIBSVM:")
    df.printSchema()
    print("Loaded training data as a DataFrame with " +
          str(df.count()) + " records.")

    # Show statistical summary of labels.
    labelSummary = df.describe("label")
    df.show()

    # Convert features column to an RDD of vectors.
    features = MLUtils.convertVectorColumnsFromML(df, "features") \
        .select("features").rdd.map(lambda r: r.features)
    summary = Statistics.colStats(features)
    print("Selected features column with average values:\n" +
          str(summary.mean()))
    spark.stop()

# (1) 尝试其他数据文件，如何观察dataframe数据的各列信息? 
