from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
from configs import d_path

# Load and parse the data
conf = SparkConf() \
    .setAppName("MovieLensALS") \
    .set("spark.executor.memory", "2g") \
    .setMaster("local[*]")
sc = SparkContext(conf=conf)
data = sc.textFile(f"{d_path}ml/ml-1m/ratings.dat")
# 用户ID， 物品ID， 得分
ratings = data.map(lambda l: l.split('::')) \
    .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
print("Mean Squared Error = " + str(MSE))

# Save and load model
model.save(sc, "target/tmp/myCollaborativeFilter")
sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")
