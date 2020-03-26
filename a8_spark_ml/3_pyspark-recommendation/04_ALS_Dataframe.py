"""
Spark DataFrame ALS
参考文档：
    https://www.cnblogs.com/pinard/p/6364932.html
"""
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, SparkSession
from configs import d_path

spark = SparkSession \
    .builder \
    .appName("ctr_sample") \
    .getOrCreate()

lines = spark.read.text(f"{d_path}ml/mllib/als/test.data").rdd
parts = lines.map(lambda row: row.value.split("::"))

# 评分矩阵 userid, moveid, score, timestampe
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=long(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics

'''---------------------
:param
    rank : 矩阵分解时对应的低维的维数，通常需要进行调参，一般可以取10-200之间的数
    iterations: 取决于评分矩阵的维度，一般来说，不需要太大，比如5-20次即可。默认值是5
    regParam: 取值越大，则正则化惩罚越强
    alpha : 这个参数仅仅在使用隐式反馈trainImplicit时有用，一般需要调参得到合适值。
        指定了隐式反馈信心阈值，这个值越大则越认为用户和他没有评分的物品之间没有关联。
-------------------------'''
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")

model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data, predict
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)

# Generate top 10 movie recommendations for a specified set of users
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
# Generate top 10 user recommendations for a specified set of movies
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)