# /opt/spark/bin/spark-submit --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.1.1 tweets_to_hdfs.py

from p3lib import *
import json, re, string, time
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from datetime import datetime, timedelta, timezone

sc = SparkContext(appName="KafkaSparkStream-p3-tweets-sentiment-tohdfs")
sc.setLogLevel("WARN")

ssc = StreamingContext(sc, 60)

kafkaStream = KafkaUtils.createStream(ssc, 'data04:2181', 'trump-consumer-group2', {'trump':1})

dataJson = kafkaStream.map(lambda x: json.loads(x[1]))
messages = dataJson.map(lambda x: (x['text'], datetime.strptime(x['created_at'], '%a %b %d %H:%M:%S %z %Y').replace(tzinfo=timezone.utc)))
messages_downsecs = messages.map(lambda x: (x[0], x[1] - timedelta(seconds=x[1].second, microseconds=x[1].microsecond)))

parts = messages_downsecs.map(lambda x: json.dumps({'tweet': x[0], 'sentence': processTweetText(x[0]), 'created_at': x[1].isoformat()}))

parts.count().map(lambda x:'Tweets in this batch: %s' % x).pprint()
parts.saveAsTextFiles('hdfs://master:9000/home/omar/p3/tweets.txt')

ssc.start()
ssc.awaitTermination()
