from django.db import models
import tweepy
from pymongo import MongoClient

def accesoYAlmacenamiento():

    consumer_key = "XDlZH2xkKbnl1lGBZfJ5DQdvg"
    consumer_secret = "TIp073JeX7EgJZZ5GVeOyrJxpltuXiErBhYwoti8ekeGy8bxQz"
    access_token = "632399680-0PflY50FysQePc87NghCd8uezrd3qrE9zZZhhKel"
    access_token_secret = "4mnR0EC5utORHOsx0XkGjRyseqRzCo7xSQDb970NodzZp"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    try:
        client = MongoClient()
        # Use twitterdb database. If it doesn't exist, it will be created.
        db = client['test']
        mycol = db['coltest']
        print("Conectado a la colección")
        for tweet in tweepy.Cursor(api.search, q="#GHVIP17O", count="2").items():
            post = tweet
        mycol.insert_one(post)

    except Exception as e:
        print(e)
