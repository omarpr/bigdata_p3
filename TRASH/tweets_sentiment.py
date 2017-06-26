# /opt/spark/bin/spark-submit --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.1.1 --conf spark.executor.heartbeatInterval=3600s TRASH/tweets_sentiment.py

import json, re, string, time
from p3lib import *
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from datetime import datetime, timedelta, timezone
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import *
from pyspark.sql.functions import rand
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
from pyspark.ml.classification import LogisticRegression

sc = SparkContext(appName="KafkaSparkStream-p3-tweets-sentiment")
sc.setLogLevel("WARN")

ssc = StreamingContext(sc, 60)
sqlContext = SQLContext(sc)

kafkaStream = KafkaUtils.createStream(ssc, 'data04:2181', 'trump-consumer-group2', {'trump':1})

dataJson = kafkaStream.map(lambda x: json.loads(x[1]))
messages = dataJson.map(lambda x: (x['text'], datetime.strptime(x['created_at'], '%a %b %d %H:%M:%S %z %Y').replace(tzinfo=timezone.utc)))
messages_downsecs = messages.map(lambda x: (x[0], x[1] - timedelta(seconds=x[1].second, microseconds=x[1].microsecond)))

parts = messages_downsecs.map(lambda x: Row(tweet=x[0], sentence=processTweetText(x[0]), created_at=x[1].isoformat()))
#parts.count().map(lambda x:'Tweets in this batch: %s' % x).pprint()

#partsDF = parts.transform(lambda rdd: rdd.toDF().collect())
#partsDF = parts.transform(lambda rdd: rdd.collect())
#partsDF = parts.transform(lambda rdd: rdd.collect())

schema = StructType([StructField('sentence', StringType(), True)])
partsDF = sqlContext.createDataFrame(sc.emptyRDD(), schema)

def RDDsToDF2(rdd, partsDF):
    print(partsDF)
    print(rdd.count())

def RDDsToDF(rdd):
    global partsDF
    print(0)
    if (rdd.count() > 0):
        if (partsDF.count() == 0):
            print(1)
            partsDF = rdd.toDF()
        else:
            print(2)
            partsDF = partsDF.unionAll(rdd.toDF())

parts.foreachRDD(RDDsToDF)

#parts.foreachRDD(lambda rdd, partsDF=partsDF: partsDF = rdd.toDF())

print('Tweets in this batchX: %x' % partsDF.count())
print(partsDF)
#partsDF.pprint()

lrModel = PipelineModel.load('/home/omar/p3/sentimentAnalysisModel')
#lrResult = lrModel.transform(partsDF.slice(0, round(datetime.utcnow().timestamp())))
lrResult = lrModel.transform(partsDF)

#lrResult.select('tweet', 'created_at', 'prediction').write.format('com.databricks.spark.csv').save('hdfs://master:9000/home/omar/p3/tweets.txt')
parts.saveAsTextFiles('hdfs://master:9000/home/omar/p3/tweets.txt')

ssc.start()
ssc.awaitTermination()
