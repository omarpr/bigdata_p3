# pyspark --master yarn --deploy-mode client --conf='spark.executorEnv.PYTHONHASHSEED=223'

import re, csv, string
from pyspark.sql import Row
from pyspark.sql.functions import rand
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
from pyspark.ml.classification import LogisticRegression

maxLines = 500000
trainingFile = "/home/omar/sentiment-train.csv" # Must be CSV. Column 4 contains text, Column 2 contains sentiment.

def processTweetText(tweet):
    new_tweet = ''
    tweet = tweet.strip().translate(str.maketrans('','',string.punctuation.replace('@','').replace('#','')))
    tweet = re.sub('\d+', '', tweet.strip())
    tweet = re.sub('\s+', ' ', tweet)
    for word in tweet.split():
        if re.match('^.*@.*', word):
            word = '<SCREENNAME/>'
        if re.match('^.*http://.*', word):
            word = '<LINK/>'
        word = word.replace('#', '<HASHTAG/> ')
        word = word.replace('&quot;', ' \" ')
        word = word.replace('&amp;', ' & ')
        word = word.replace('&gt;', ' > ')
        word = word.replace('&lt;', ' < ')
        new_tweet = ' '.join([new_tweet, word])
    return new_tweet.strip()

data = sc.textFile(trainingFile)

header = data.first()
rdd = data.filter(lambda row: row != header)

r = rdd.mapPartitions(lambda x : csv.reader(x))
r2 = r.map(lambda x: (processTweetText(x[3]), int(x[1])))

parts = r2.map(lambda x: Row(sentence=x[0], label=int(x[1])))
partsDF = spark.createDataFrame(parts).orderBy(rand()).limit(maxLines)

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="base_words")

hashingTF = HashingTF(inputCol="base_words", outputCol="features")
lr = LogisticRegression(maxIter=100, regParam=0.001, elasticNetParam=0.0001)
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, lr])

lrModel = pipeline.fit(partsDF)
lrResult = lrModel.transform(partsDF)

avg = lrResult.where('label == prediction').count() / maxLines
print(avg) #0.996
