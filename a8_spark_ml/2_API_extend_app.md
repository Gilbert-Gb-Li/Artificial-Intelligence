# For RDD

```
lines = sc.textFile("data.txt")
lineLengths = lines.map(lambda s: len(s))
totalLength = lineLengths.reduce(lambda a, b: a + b)
```
### 1 map
所有不需要聚合，只需要本地运算的操作。

1.1 filter  
    过滤异常样本
1.2 map     
1.2.1 删除行（过滤异常样本） 等价filter          
1.2.2 删除异常列（空值过多）       
1.2.3 ETL 字符串预处理（通过正则表达式等）      

1.3 自定义特征工程处理       
1.3.1 空值填充。     
靠reduce先统计，再map空值填充         
1.3.2 onehot encoder。           
靠reduce先统计，再map进行字典替换。          
1.3.3 归一化           
map直接对每个样本（行）进行归一化。         

1.3.4 min-max scaler / standardscaler           
靠reduce先统计，再map进行每个样本特定列的规范化。           

1.3.5 特征选择          
reduce先统计，及某特征和y进行map运算算皮尔逊相关系数。        

或直接通过GDBT等算出特征排名。           

1.3.6 特征扩展      

特征组合            
reduce统计字典，然后靠map进行组合出离散新特征。        

### 2 reduce
2.1 统计全局列值          

2.2 统计某列以key为分组的聚合值，类似group by      

2.3 辅助其他数据处理（特征工程）或算法，收集全局数据或字典     

### 3 数据共享 cross jobs       

3.0 spark inner shuffle for kv between MR       
3.1 collect & broadcast.        
3.2 HDFS or other storages.     

# For dataframe     

# PySpark特征工程
1. PySpark RDD与Dataframe转换:     
    1_pyspark_dataframe/ml/0_rdd_dataframe_convert.py
- Dataframe有封装好的feature engineering API开发效率高。
- RDD可以使用map等lamda表达式，灵活性强。

2. Dataframe API与RDD API混合编程        
3_pyspark-recommendation/ALS_Dataframe.py       
[Ref doc](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)

3. Dataframe特征工程API

1_pyspark_dataframe 文件夹

4. RDD map / filter lamda表达式 and reduce 

2_spark_kdd19/nb-rdd-basics/nb2-rdd-basics.py

2_spark_kdd19/nb5-rdd-aggregations/nb5-rdd-aggregations.py

## PySpark机器学习

### API区分

#### 引用区分
RDD-based:
from pyspark.mllib

Dataframe-based:
from pyspark.ml

#### 数据读入API区分

RDD-based:

```
# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile("data/mllib/sample_svm_data.txt")
parsedData = data.map(parsePoint)
```

Dataframe-based:

```
training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
```

#### 模型输入数据格式区分

RDD-based:

ML model input datatype LabelPoint

2_spark_kdd19/spark-py-notebooks-master/nb8-mllib-logit/nb8-mllib-logit.py

[Ref Doc](https://spark.apache.org/docs/latest/mllib-linear-methods.html#linear-least-squares-lasso-and-ridge-regression)


Dataframe-based:

ML model input datatype Dataframe

1_pyspark_dataframe/ml/3_logistic_regression_summary_example.py

[Ref Doc](https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression)


### 回归问题   
RDD-based:
Dataframe-based:
### 分类问题
RDD-based:
Dataframe-based:
### 聚类
RDD-based:
Dataframe-based:
### 推荐
RDD-based:      
参考实例:       
- *CF*:       
    3_pyspark-recommendation/ALS.py     
- *item-based CF*:      
    3_pyspark-recommendation/itemBasedRecommender.py
- *user-based CF*:        
    3_pyspark-recommendation/userBasedRecommender.py

Dataframe-based:    
https://spark.apache.org/docs/latest/ml-collaborative-filtering.html
