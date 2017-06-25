import re, csv, string
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

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

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))
