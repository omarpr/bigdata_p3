from kafka import KafkaProducer
from twitter import *
import json, sys

def read_credentials():
    file_name = 'twitter_credentials.json'
    try:
        with open(file_name) as data_file:
            return json.load(data_file)
    except:
        print ('Cannot load twitter_credentials.json')
        return None

credentials = read_credentials()

if (credentials == None):
    sys.exit('No twitter credentials available.')

producer = KafkaProducer(bootstrap_servers='data04:9092')
twitter_stream = TwitterStream(auth=OAuth(credentials['ACCESS_TOKEN'], credentials['ACCESS_SECRET'], credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET']))
iterator = twitter_stream.statuses.sample()

for tweet in iterator:
    if('delete' not in tweet and 'text' in tweet and 'trump' in tweet['text'].lower()):
        tweet_json = json.dumps(tweet)
        producer.send('trump', str.encode(tweet_json))
