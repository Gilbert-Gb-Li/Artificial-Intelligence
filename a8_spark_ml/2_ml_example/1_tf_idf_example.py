from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import HashingTF,IDF,Tokenizer
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("OneHotEncoderExample")\
        .getOrCreate()
    sentenceData = spark.createDataFrame([
        (0, "I heard about Spark and I love Spark"),
        (0, "I wish Java could use case classes"),
        (1, "Logistic regression models are neat")]).toDF("label", "sentence")

    '''`1. 英文分词 '''
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    wordsData = tokenizer.transform(sentenceData)

    '''2.1 TF'''
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=2000)
    featurizedData = hashingTF.transform(wordsData)
    '''
    输出解释
    # print(featurizedData.select("rawFeatures").show(1))
    # 2000: 为使用的hash桶数，较大的hash桶数可以防止hash碰撞
    # [65, 618, 852, 992, 1194]: 分别代表i, spark, heard, about, and, love的hash值
    # [1.0, 1.0, 1.0, 1.0, 1.0]: 分别代表i, spark, heard, about, and, love的出现次数
    # (2000, [65, 618, 852, 992, 1194], [1.0, 1.0, 1.0, 1.0, 1.0])]
    '''
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("label", "features").show()
    '''
    输出格式: [(feature),label]
    # [(2000,[105,365,1329,1469,1926],
    # [0.28768207245178085,0.6931471805599453,0.6931471805599453,0.6931471805599453,0.6931471805599453])
    # ,0]
    # [0.28768207245178085,0.6931471805599453,0.6931471805599453,0.6931471805599453,0.6931471805599453]
        表示i, spark, heard, about, and, love的tf-idf值。
    '''
    data = rescaledData.select("label", "features")
    '''
    # 使用随机森林训练
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
    rf.fit(dataset=data)
    '''
