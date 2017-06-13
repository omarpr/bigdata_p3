# pyspark --master yarn --deploy-mode client --conf='spark.executorEnv.PYTHONHASHSEED=223'

import re, csv, string
from pyspark.sql import Row
from pyspark.sql.functions import rand
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
from pyspark.ml.classification import LogisticRegression

trainPercent = 0.6
testPercent = 1 - trainPercent
maxLines = 500000
trainingFile = "/home/omar/sentiment-train.csv" # Must be CSV. Column 4 contains text, Column 2 contains sentiment.

def processTweetText(tweet):
    tweet = tweet.strip()
    tweet = re.sub(r'[^\s]*htt(p|ps)://[^\s]*', 'LINK', tweet)
    tweet = re.sub(r'@[^\s]*', 'SCREENNAME', tweet)
    tweet = tweet.replace('#', 'HASHTAG ').replace('&quot;', ' \" ').replace('&amp;', ' & ').replace('&gt;', ' > ').replace('&lt;', ' < ')
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet = re.sub('\d+', '', tweet)
    tweet = re.sub('\s+', ' ', tweet)
    return tweet.strip()

data = sc.textFile(trainingFile)

header = data.first()
rdd = data.filter(lambda row: row != header)

r = rdd.mapPartitions(lambda x : csv.reader(x))
r2 = r.map(lambda x: (processTweetText(x[3]), int(x[1])))

parts = r2.map(lambda x: Row(sentence=x[0], label=int(x[1])))
partsDF = spark.createDataFrame(parts).orderBy(rand()).limit(maxLines)

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="base_words")
hashingTF = HashingTF(numFeatures=10000, inputCol="base_words", outputCol="features")

lr = LogisticRegression(maxIter=10000, regParam=0.001, elasticNetParam=0.0001)

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, lr])

splits = partsDF.randomSplit([trainPercent, testPercent], 1291)
trainSet = splits[0]
testSet = splits[1]

lrModel = pipeline.fit(trainSet)
lrResult = lrModel.transform(testSet)

avg = lrResult.where('label == prediction').count() / (maxLines * testPercent)
print(avg)
