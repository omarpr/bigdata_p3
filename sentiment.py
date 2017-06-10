# pyspark --master yarn --deploy-mode client --conf='spark.executorEnv.PYTHONHASHSEED=223'

import csv
from pyspark.sql import Row
from pyspark.sql.functions import rand
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import LogisticRegression

#data = sc.textFile("/home/omar/sentiment-train.csv")
data = sc.textFile("/home/omar/sentiment-test.csv")
#data = sc.textFile("/home/omar/sentiment-manuel.csv")
header = data.first()
rdd = data.filter(lambda row: row != header)

r = rdd.mapPartitions(lambda x : csv.reader(x))
r2 = r.map(lambda x: (x[3], int(x[1])))

parts = r2.map(lambda x: Row(sentence=x[0], label=int(x[1])))
partsDF = spark.createDataFrame(parts)
partsDF = partsDF.orderBy(rand()).limit(10000)

#partsDF.show(truncate=False)

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
tokenized = tokenizer.transform(partsDF)

#tokenized.show(truncate=False)

remover = StopWordsRemover(inputCol="words", outputCol="base_words")
base_words = remover.transform(tokenized)

#base_words.show(truncate=False)

train_data_raw = base_words.select("base_words", "label")

#train_data_raw.show(truncate=False)

word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol="base_words", outputCol="features")
#word2Vec = CountVectorizer(inputCol="base_words", outputCol="features")

model = word2Vec.fit(train_data_raw)
final_train_data = model.transform(train_data_raw)
final_train_data = final_train_data.select("label", "features")

#final_train_data.show(truncate=False)

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(final_train_data)

lrModel.transform(final_train_data).show()

#t = lrModel.transform(final_train_data)
#t.where('label != prediction').show();
#t.select('probability').groupBy('probability').count()
