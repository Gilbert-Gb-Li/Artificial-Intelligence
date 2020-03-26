from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .getOrCreate()

sum = spark.sparkContext\
    .textFile('data/*')\
    .flatMap(lambda x: x.split(','))\
    .map(lambda x: int(x)).reduce(lambda x, y: x + y)

print(sum)

spark.stop()