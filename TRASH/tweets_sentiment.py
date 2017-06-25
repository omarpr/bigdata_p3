# /opt/spark/bin/spark-submit --packages org.apache.spark:spark-streaming-kaf-0-8_2.11:2.1.1 tweets_sentiment.py

import json
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from datetime import datetime, timedelta, timezone
from pyspark.sql import SQLContext, Row
from pyspark.sql.functions import rand
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
from pyspark.ml.classification import LogisticRegression

def processTweetText(tweet):
    tweet = tweet.strip()
    tweet = re.sub(r'[^\s]*htt(p|ps)://[^\s]*', 'LINK', tweet)
    tweet = re.sub(r'@[^\s]*', 'SCREENNAME', tweet)
    tweet = tweet.replace('#', 'HASHTAG ').replace('&quot;', ' \" ').replace('&amp;', ' & ').replace('&gt;', ' > ').replace('&lt;', ' < ')
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet = re.sub('\d+', '', tweet)
    tweet = re.sub('\s+', ' ', tweet)
    return tweet.strip()

sc = SparkContext(appName="KafkaSparkStream-p2-screennames")
sc.setLogLevel("WARN")

ssc = StreamingContext(sc, 60)
sqlContext = SQLContext(sc)

kafkaStream = KafkaUtils.createStream(ssc, 'data04:2181', 'trump-consumer-group2', {'trump':1})

dataJson = kafkaStream.map(lambda x: json.loads(x[1]))
messages = dataJson.map(lambda x: (x['text'], datetime.strptime(x['created_at'], '%a %b %d %H:%M:%S %z %Y').replace(tzinfo=timezone.utc)))
messages_downsecs = messages.map(lambda x: (x[0], x[1] - timedelta(seconds=x[1].second, microseconds=x[1].microsecond)))

parts = messages_downsecs.map(lambda x: Row(tweet=x[0], sentence=processTweetText(x[0]), created_at=x[1].isoformat()))

#rdd = parts.foreachRDD(lambda x: x)
#partsDF = sqlContext.createDataFrame(parts)
partsDF = parts.toDF()

lrModel = PipelineModel.load('/home/omar/p3/sentimentAnalysisModel')
lrResult = lrModel.transform(partsDF)

lrResult.count().map(lambda x:'Tweets in this batch: %s' % x).pprint()

lrResult.select('tweet', 'created_at', 'prediction').write.format('com.databricks.spark.csv').save('hdfs://master:9000/home/omar/p3/tweets.txt')

ssc.start()
ssc.awaitTermination()
