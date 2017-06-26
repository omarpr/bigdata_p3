# /opt/spark/bin/spark-submit --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.1.1 --conf spark.executor.heartbeatInterval=3600s build_model.py

from p3lib import *
import re, csv, string
from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import rand
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
from pyspark.ml.classification import LogisticRegression

trainPercent = 0.6
testPercent = 1 - trainPercent
maxLines = 500000
trainingFile = '/home/omar/sentiment-train.csv' # Must be CSV. Column 4 contains text, Column 2 contains sentiment.
modelFile = '/home/omar/p3/sentimentAnalysisModel'

sc = SparkContext(appName="KafkaSparkStream-p3-build_model")
sc.setLogLevel("WARN")

spark = SparkSession(sc)

data = sc.textFile(trainingFile)

header = data.first()
rdd = data.filter(lambda row: row != header)

r = rdd.mapPartitions(lambda x : csv.reader(x))
r2 = r.map(lambda x: (processTweetText(x[3]), int(x[1])))

parts = r2.map(lambda x: Row(sentence=x[0], label=int(x[1])))
partsDF = spark.createDataFrame(parts).orderBy(rand()).limit(maxLines)

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="base_words")
hashingTF = HashingTF(numFeatures=6000, inputCol="base_words", outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.05, elasticNetParam=0.025, family="binomial")

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, lr])

(trainSet, testSet) = partsDF.randomSplit([trainPercent, testPercent], 1291)

lrModel = pipeline.fit(trainSet)
lrResult = lrModel.transform(testSet)

avg = lrResult.where('label == prediction').count() / (maxLines * testPercent)
print(avg) # 0.752015

lrModel.write().overwrite().save(modelFile)
