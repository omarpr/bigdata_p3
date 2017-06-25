# pyspark --master yarn --deploy-mode client --conf='spark.executorEnv.PYTHONHASHSEED=223'

import re
import csv
from pyspark.sql import Row
from pyspark.sql.functions import rand
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Word2Vec
from pyspark.ml.classification import LogisticRegression, LogisticRegressionWithLBFGS
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import *

maxLines = 10000

data = sc.textFile("/home/omar/sentiment-train.csv")

header = data.first()
rdd = data.filter(lambda row: row != header)

r = rdd.mapPartitions(lambda x : csv.reader(x))
r2 = r.map(lambda x: (re.sub('\s+', ' ', x[3]).strip(), int(x[1])))

parts = r2.map(lambda x: Row(sentence=x[0], label=int(x[1])))
partsDF = spark.createDataFrame(parts)
partsDF = partsDF.orderBy(rand()).limit(maxLines)

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
tokenized = tokenizer.transform(partsDF)

remover = StopWordsRemover(inputCol="words", outputCol="base_words")
base_words = remover.transform(tokenized)

train_data_raw = base_words.select("label", "base_words")

word2Vec = Word2Vec(vectorSize=100, minCount=100, inputCol="base_words", outputCol="features")

model = word2Vec.fit(train_data_raw)
final_train_data = model.transform(train_data_raw)
final_train_data = final_train_data.select("label", "features")

parsedData = final_train_data.rdd.map(lambda x: LabeledPoint(x.label, x.features.toArray()))
#model = SVMWithSGD.train(parsedData, iterations=1000, step=0.01, regParam=0.001)
model = LogisticRegressionWithSGD.train(parsedData, maxIter=100, regParam=0.001, elasticNetParam=0.0001)

train(cls, data, iterations=100, step=1.0, miniBatchFraction=1.0, initialWeights=None, regParam=1.0, regType=None, intercept=False)

labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda x: x[0] != x[1]).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

import matplotlib.pyplot as plt
plt.plot(model.summary.objectiveHistory)
plt.show()

#Training Error = 0.325388

#avg = lrResult.where('label == prediction').count() / maxLines
#print(avg)

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [10, 100, 1000]) \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=3)  # use 3+ folds in practice

cvModel = crossval.fit(trainSet)

prediction = cvModel.transform(testSet)
#selected = prediction.select("id", "text", "probability", "prediction")
for row in prediction.collect():
    print(row)
