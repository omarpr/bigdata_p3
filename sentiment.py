# pyspark --master yarn --deploy-mode client --conf='spark.executorEnv.PYTHONHASHSEED=223'

import re
import csv
from pyspark.sql import Row
from pyspark.sql.functions import rand
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Word2Vec
from pyspark.ml.classification import LogisticRegression

def processTweetText(tweet):
    new_tweet = ''

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

    return re.sub('\s+', ' ', new_tweet.strip())

maxLines = 500000

data = sc.textFile("/home/omar/sentiment-train.csv")

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

word2Vec = Word2Vec(vectorSize=100, minCount=100, inputCol="base_words", outputCol="features")

model = word2Vec.fit(train_data_raw)
final_train_data = model.transform(train_data_raw)
final_train_data = final_train_data.select("label", "features")

#lr = LogisticRegression(maxIter=100, regParam=0.1, elasticNetParam=0.8) # Success: 0.642788
#lr = LogisticRegression(maxIter=100, regParam=0.1, elasticNetParam=0.6) # Success: 0.669502
#lr = LogisticRegression(maxIter=100, regParam=0.03, elasticNetParam=0.4) # Success: 0.718
#lr = LogisticRegression(maxIter=100, regParam=0.003, elasticNetParam=0.001) # Success: 0.728092
lr = LogisticRegression(maxIter=1000, regParam=0.001, elasticNetParam=0.0001) # Success: 0.728234
lrModel = lr.fit(final_train_data)
lrResult = lrModel.transform(final_train_data)

avg = lrResult.where('label == prediction').count() / maxLines
print(avg)

lrResult.show()
