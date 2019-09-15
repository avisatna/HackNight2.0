from django.http import HttpResponse
from django.shortcuts import render, render_to_response
from django.contrib import messages
import re 
import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob 

"""class TwitterClient(object):
    def __init__(self):
        consumer_key = 'OZxUHnjfaGs54tQTEBROmQ6oP'
        consumer_secret = 'k2yegjgGb3Q48oYs2sEYmH6cGE2j24UEtjeeSaAGzCrkoq1MZS'
        access_token = '122644705-2XFloUaPy9wvuwhitHvqsFDO9gig41KT2UuYf1GZ'
        access_token_secret = 'lBmtVCCo8fxHHMfQtPoJth1bihOhxKOTqzcDAbLPGXQsz'
        try: 
            # create OAuthHandler object 
            self.auth = OAuthHandler(consumer_key, consumer_secret) 
            # set access token and secret 
            self.auth.set_access_token(access_token, access_token_secret) 
            # create tweepy API object to fetch tweets 
            self.api = tweepy.API(self.auth) 
        except: 
            print("Error: Authentication Failed") 
  
    def clean_tweet(self, tweet): 
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
        
    def get_tweet_sentiment(self, tweet): 
        # create TextBlob object of passed tweet text 
        analysis = TextBlob(self.clean_tweet(tweet)) 
        # set sentiment 
        if analysis.sentiment.polarity > 0: 
            return 'positive'
        elif analysis.sentiment.polarity == 0: 
            return 'neutral'
        else: 
            return 'negative'
            
    def get_tweets(self, query, count = 10): 
        ''' 
        Main function to fetch tweets and parse them. 
        '''
        # empty list to store parsed tweets 
        tweets = [] 
  
        try: 
            # call twitter api to fetch tweets 
            fetched_tweets = self.api.search(q = query, count = count) 
  
            # parsing tweets one by one 
            for tweet in fetched_tweets: 
                # empty dictionary to store required params of a tweet 
                parsed_tweet = {} 
  
                # saving text of tweet 
                parsed_tweet['text'] = tweet.text 
                # saving sentiment of tweet 
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text) 
  
                # appending parsed tweet to tweets list 
                if tweet.retweet_count > 0: 
                    # if tweet has retweets, ensure that it is appended only once 
                    if parsed_tweet not in tweets: 
                        tweets.append(parsed_tweet) 
                else: 
                    tweets.append(parsed_tweet) 
  
            # return parsed tweets 
            return tweets 
  
        except tweepy.TweepError as e: 
            # print error (if any) 
            print("Error : " + str(e)) """

# Create your views here.
def index(request):
    if request.method == "GET":
        """api = TwitterClient()
        # calling function to get tweets
        tweets = api.get_tweets(query = 'Samsung electronics', count = 200) 
  
        # picking positive tweets from tweets 
        ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive'] 
        # percentage of positive tweets 
        messages.success(request, "Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets))) 
        # picking negative tweets from tweets 
        ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
        # percentage of negative tweets 
        messages.success(request, "Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets))) 
        # percentage of neutral tweets 
        messages.success(request, "Neutral tweets percentage: {} % ".format(100*len(tweets - ntweets - ptweets)/len(tweets))) """
  
        return render(request, "index.html", {})