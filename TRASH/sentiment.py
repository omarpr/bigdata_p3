# pyspark --master yarn --deploy-mode client --conf='spark.executorEnv.PYTHONHASHSEED=223'

import re, csv, string
import matplotlib.pyplot as plt
from pyspark.sql import Row
from pyspark.sql.functions import rand
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec, HashingTF
from pyspark.ml.classification import LogisticRegression

maxLines = 5000
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
partsDF = spark.createDataFrame(parts)
partsDF = partsDF.orderBy(rand()).limit(maxLines)

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
tokenized = tokenizer.transform(partsDF)

remover = StopWordsRemover(inputCol="words", outputCol="base_words")
base_words = remover.transform(tokenized)

train_data_raw = base_words.select("base_words", "label")




hashingTF = HashingTF(inputCol="base_words", outputCol="features")
lr = LogisticRegression(maxIter=100, regParam=0.001, elasticNetParam=0.0001)
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, lr])

lrModel = pipeline.fit(partsDF)
lrResult = lrModel.transform(final_train_data)







word2Vec = Word2Vec(vectorSize=100, minCount=100, inputCol="base_words", outputCol="features")

model = word2Vec.fit(train_data_raw)
final_train_data = model.transform(train_data_raw)
final_train_data = final_train_data.select("label", "features")

lr = LogisticRegression(maxIter=100, regParam=0.001, elasticNetParam=0.0001, family="binomial") # Success: 0.728234
lrModel = lr.fit(final_train_data)
lrResult = lrModel.transform(final_train_data)



##TEST
##https://stackoverflow.com/questions/28256058/plotting-decision-boundary-of-logistic-regression
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white")

xx, yy = np.mgrid[0:1:.01, 0:1:.01]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = lrModel.predict_proba(grid)[:, 1].reshape(xx.shape)

##TEST
#plt.plot(lrModel.summary.objectiveHistory)
#plt.show()

lrModel.weights
lrModel.intercept

avg = lrResult.where('label == prediction').count() / maxLines
print(avg)

lrResult.show()
