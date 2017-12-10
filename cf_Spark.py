from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession, Row, functions, Column
from pyspark.sql.types import *

def date2int(date):
	l = date.split('-')
	return int(l[0]+l[1]+l[2])

spark = SparkSession.builder.appName('cf').getOrCreate()

input_file='/home/ehsan/WIP/DataMininig/train_rating.txt'
input_file='train_rating.txt'
schema = StructType([\
StructField('train_id',IntegerType(), False),\
StructField('user_id', IntegerType(), False),\
StructField('business_id', IntegerType(), False),\
StructField('rating', IntegerType(), False),\
StructField('date', StringType(), False)])

lines = spark.read.csv(input_file,schema=schema,sep=',')
lines = lines.drop('train_id').na.drop()

date2intUDF = udf(date2int,IntegerType())
lines = lines.withColumn('date',date2intUDF('date')) #Converts the date from string to dateType


#lines = lines.rdd
#parts = lines.map(lambda row: row.value.split("::"))
#ratingsRDD = parts.map(lambda p: Row(user_id=int(p[1]), business_id=int(p[2]),rating=float(p[3]), date=long(p[4])))
#ratings = spark.createDataFrame(ratingsRDD)
#(training, test) = ratings.randomSplit([0.8, 0.2])
(training, test) = lines.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
als = ALS(maxIter=10, rank=100,regParam=0.01, userCol="user_id", itemCol="business_id",ratingCol="rating",coldStartStrategy='drop',implicitPrefs=False)

model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)

minVal = predictions.groupBy().min('prediction').collect()[0]['min(prediction)']
maxVal = predictions.groupBy().max('prediction').collect()[0]['max(prediction)']

predictions = predictions.withColumn('predNorm',1+4*( (predictions['prediction']-minVal)/(maxVal-minVal) ))


evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="predNorm")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

evaluator = RegressionEvaluator(metricName="mae", labelCol="rating",
                                predictionCol="predNorm")
rmse = evaluator.evaluate(predictions)
print("MAE = " + str(rmse))
