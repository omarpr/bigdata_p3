# pyspark --master yarn --deploy-mode client --conf='spark.executorEnv.PYTHONHASHSEED=223'

from p3lib import *
import json, string, numpy
from pyspark.sql import Row, SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import PipelineModel

main_dir = '/home/omar.soto2/p3/files/'
words = ['maga', 'dictator', 'impeach', 'drain', 'swamp', 'comey']

SparkContext.setSystemProperty('spark.executor.memory', '3G')

sc = SparkContext(appName="p3-tweets_sentiment-cron")
sqlContext = SQLContext(sc)
sc.setLogLevel("WARN")

a = sc.textFile('/home/omar/p3/tweets.txt-*/part-*')
b = a.map(lambda x: json.loads(x))
c = b.map(lambda x: Row(tweet=x['tweet'], created_at=x['created_at'], sentence=x['sentence']))

df = sqlContext.createDataFrame(c)
ndf = df.withColumn('created_at', df['created_at'].cast(TimestampType()))

ndf.registerTempTable('tweets')

wheres = []
for x in words:
    wheres.append('lower(sentence) like "%' + x + '%"')

df_select = sqlContext.sql('SELECT * FROM tweets WHERE ' +  ' OR '.join(wheres))

lrModel = PipelineModel.load('/home/omar/p3/sentimentAnalysisModel')
lrResult = lrModel.transform(df_select)
rdd = lrResult.rdd

resultToFiles(rdd, main_dir, 'tweet_sentiments', '1h')
