# -*- coding:utf-8 -*- 
import os
import sys
# Path for spark source folder
# os.environ['SPARK_HOME']="D:\\data_and_dep\\sparkml\\software\\spark-2.2.0-bin-hadoop2.7\\spark-2.2.0-bin-hadoop2.7"
# os.environ['JAVA_HOME']="C:\\Program Files\\Java\\jre1.8.0_151"
# os.environ['HADOOP_HOME']="D:\\data_and_dep\\sparkml\\software\\spark-2.2.0-bin-hadoop2.7\\spark-2.2.0-bin-hadoop2.7"
# # # Append pyspark  to Python Path
# sys.path.append("D:\data_and_dep\sparkml\software\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7\python")
# sys.path.append("D:\data_and_dep\sparkml\software\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7\python\lib/py4j-0.9-src.zip")
from pyspark import SparkContext, SparkConf
from operator import add

sc = SparkContext("local[*]")  # SparkSession

# RDD
lines = sc.textFile("0_test/test.txt")


def reduce_func(a, b):
    return a + b


def flat_func(x):
    return x.split(' ')


# 分布式的RDD上执行的
tmp = lines.flatMap(lambda x: flat_func(x))

lines = sc.textFile("test.txt")
tmp = lines.flatMap(lambda x: x.split(' ')).map(lambda x: (x, 1))
counts = tmp.reduceByKey(reduce_func)  # group by key
output = counts.collect()  # count / take  === action
# counts.count()
for (word, count) in output:
    print("xxx: %s %i" % (word, count))

print(counts)

sc.stop()
