# pyspark --master yarn --deploy-mode client --conf='spark.executorEnv.PYTHONHASHSEED=223'
# /opt/spark/bin/spark-submit --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.1.1 tweets_sentiment-cron.py

from p3lib import *
import json, string, time, numpy
from pyspark import SparkContext
from pyspark.sql import Row, SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
from datetime import timedelta, datetime

main_dir = '/home/omar.soto2/p3/files/'
words = ['maga', 'dictator', 'impeach', 'drain', 'swamp', 'comey']

SparkContext.setSystemProperty('spark.executor.memory', '3G')

sc = SparkContext(appName='p3-tweets_sentiment-cron')
sqlContext = SQLContext(sc)
sc.setLogLevel('WARN')

a = sc.textFile('/home/omar/p3/tweets.txt-*/part-*')
b = a.map(lambda x: json.loads(x))
c = b.map(lambda x: (x['tweet'], (datetime.strptime(x['created_at'].split('+')[0], '%Y-%m-%dT%H:%M:%S') - timedelta(seconds=time.timezone)), x['sentence']))
d = c.map(lambda x: Row(tweet=x[0], created_at=x[1] - timedelta(minutes=x[1].minute), sentence=x[2]))

df = sqlContext.createDataFrame(d)
df.withColumn('created_at', df['created_at'].cast(TimestampType())).registerTempTable('tweets')

wheres = []
for x in words:
    wheres.append('lower(sentence) like "%' + x + '%"')

df_select = sqlContext.sql('SELECT * FROM tweets WHERE created_at >= (now() - interval 3 hours) AND (' +  ' OR '.join(wheres) + ')')

lrModel = PipelineModel.load('/home/omar/p3/sentimentAnalysisModel')
lrResult = lrModel.transform(df_select)
rdd = lrResult.rdd

resultToFiles(rdd, main_dir, 'tweet_sentiments', '1h')
