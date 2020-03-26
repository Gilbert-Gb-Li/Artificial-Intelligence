from pyspark import SparkContext 
import numpy as np

sc = SparkContext()

text_file = sc.textFile("test.txt")

def map_local_func(item):
    result = []
    for i in item:
        result.append((i, 1))
    return result

def reduce_local_func(a, b):
    return a + b

counts1 = text_file.map(map_local_func) \
             .reduce(lambda a, b: reduce_local_func(a, b))

counts2 = text_file.mapPartitions(map_local_func) \
             .reduce(lambda a, b: reduce_local_func(a, b))
             
print(counts1)

print(counts2)

