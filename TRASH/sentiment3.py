# pyspark --master yarn --deploy-mode client --conf='spark.executorEnv.PYTHONHASHSEED=223'

import re, csv, string
from pyspark.sql import Row
from pyspark.sql.functions import rand
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier

trainPercent = 0.6
testPercent = 1 - trainPercent
maxLines = 500000
trainingFile = "/home/omar/training.1600000.processed.noemoticon.csv"

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
r2 = r.map(lambda x: (processTweetText(x[5]), int(x[0])))

parts = r2.map(lambda x: Row(sentence=x[0], label=int(x[1])))
partsDF = spark.createDataFrame(parts).orderBy(rand()).limit(maxLines)

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="base_words")
hashingTF = HashingTF(numFeatures=10000, inputCol="base_words", outputCol="features")
#classifier = LogisticRegression(maxIter=10000, regParam=0.001, elasticNetParam=0.0001)
classifier = DecisionTreeClassifier(labelCol="label", featuresCol="features")

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, classifier])

(trainSet, testSet) = partsDF.randomSplit([trainPercent, testPercent], 1291)

lrModel = pipeline.fit(trainSet)
lrResult = lrModel.transform(testSet)

avg_neg = lrResult.where('label == prediction and label == 0').count() / lrResult.where('label == 0').count()
avg_neu = lrResult.where('label == prediction and label == 2').count() / lrResult.where('label == 2').count()
avg_pos = lrResult.where('label == prediction and label == 4').count() / lrResult.where('label == 4').count()

print("Negative Success Rate: " + avg_neg)
print("Neutral Success Rate: " + avg_neu)
print("Positive Success Rate: " + avg_pos)

#avg = lrResult.where('label == prediction').count() / (maxLines * testPercent)
#print(avg)
