# Big Data Analytics (CIIC 8995) Project II
Project II of Big Data Analytics (CIIC 8995) given by Dr. Manuel Rodríguez in the University of Puerto Rico, Mayagüez Campus.

An example of this is available at http://kvm_33.uprm.edu/p3/.

## twitter_stream.py
Python Kafka Producer that receives a sample stream of tweets from Twitter, extract only the ones from about *trump* and send it to the Kafka Server. A credential file is required as *twitter_credentials.json*. A example of that file is included as *twitter_credentials.sample.json*.

```shell
python3 twitter_stream.py
```

## p3lib.py
Library with the common functions for this project.

## build_model.py
Implementation of LogisticRegression using Spark Machine Learning Library (MLlib) to create a model and validate its accuracy. Later, that model is stored on HDFS to be later used to do sentiment analysis of tweets.

```shell
/opt/spark/bin/spark-submit --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.1.1 build_model.py
```

## tweets_to_hdfs.py
Python Kafka Consumer that implements Spark Streams, it receives the json of tweets about *trump*, process the tweet and store it on HDFS to be later analyzed.

```shell
/opt/spark/bin/spark-submit --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.1.1 tweets_to_hdfs.py
```

## tweets_sentiment-cron.py
Take as input the files generated by *tweets_to_hdfs.py* (and stored in HDFS) to generate the index and files that will be used by the webapp to visualize the data.

## Crontab
A crontab was configured to execute *tweets_sentiment-cron.py* every 10 minutes. The files produced by that process are then used to be visualized on a webapp.

```shell
*/10 * * * * source /home/omar.soto2/.bash_profile; flock -w 0 /home/omar.soto2/p3/tweets_sentiment-cron.lock /opt/spark/bin/spark-submit /home/omar.soto2/p3/tweets_sentiment-cron.py >> /home/omar.soto2/p3/cron_log 2>&1
```
