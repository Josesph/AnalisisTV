from django.shortcuts import render
import json
from pandas import DataFrame
import pandas as pd
import numpy as np
import nltk
from string import punctuation
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from bokeh.plotting import figure, output_file, show
from bokeh.embed import components
import sys
import string
import operator
import collections
import matplotlib.pyplot as plt
from bokeh.plotting import *
from numpy import pi
from pymongo import MongoClient
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob

def principal(request):
    mongo_test()
    id = request.GET.get('id')
    print(id)
    df, nombrePrograma = leerTuits(id)

    RTmax = topRT(df)
    FAVmax = topFAV(df)
    HTmax = topHT(df)
    script4, div4 = pieChart(df)
    script5, div5 = Polaridad(df)
    MENmax = topMEN(df)
    script1, div1 = contarPalabras(df)
    script2, div2 = tuitsPorHora(df)
    script3, div3 = tuitsPorDia(df)

    MEN1, MEN1n, MEN2, MEN2n, MEN3, MEN3n, MEN4, MEN4n, MEN5, MEN5n = allMEN(df)
    HT1, HT1n, HT2, HT2n, HT3, HT3n, HT4, HT4n, HT5, HT5n = allHT(df)
    FAV1t, FAV1n, FAV2t, FAV2n, FAV3t, FAV3n, FAV4t, FAV4n, FAV5t, FAV5n, FAV6t, FAV6n, FAV7t, FAV7n, FAV8t, FAV8n, FAV9t, FAV9n, FAV10t, FAV10n  = allFAV(df)
    RT1t, RT1n, RT2t, RT2n, RT3t, RT3n, RT4t, RT4n, RT5t, RT5n, RT6t, RT6n, RT7t, RT7n, RT8t, RT8n, RT9t, RT9n, RT10t, RT10n = allRT(df)

    contexto = {'script1': script1, 'div1': div1, 'script2':script2, 'div2':div2, 'RTmax':RTmax, 'FAVmax':FAVmax,
                'HTmax':HTmax, 'MENmax':MENmax, 'script3':script3, 'div3':div3, 'FAV1t': FAV1t, 'FAV1n': FAV1n,
                'FAV2t': FAV2t, 'FAV2n': FAV2n, 'FAV3t': FAV3t, 'FAV3n': FAV3n, 'FAV4t': FAV4t, 'FAV4n': FAV4n,
                'FAV5t': FAV5t, 'FAV5n': FAV5n, 'FAV6t': FAV6t, 'FAV6n': FAV6n, 'FAV7t': FAV7t, 'FAV7n': FAV7n,
                'FAV8t': FAV8t, 'FAV8n': FAV8n, 'FAV9t': FAV9t, 'FAV9n': FAV9n, 'FAV10t': FAV10t, 'FAV10n': FAV10n,
                'RT1t': RT1t, 'RT1n': RT1n, 'RT2t': RT2t, 'RT2n': RT2n, 'RT3t': RT3t, 'RT3n': RT3n, 'RT4t': RT4t,
                'RT4n': RT4n, 'RT5t': RT5t, 'RT5n': RT5n, 'RT6t': RT6t, 'RT6n': RT6n, 'RT7t': RT7t, 'RT7n': RT7n,
                'RT8t': RT8t, 'RT8n': RT8n, 'RT9t': RT9t, 'RT9n': RT9n,'RT10t': RT10t, 'RT10n': RT10n,
                'nombrePrograma': nombrePrograma, 'script4': script4, 'div4': div4, 'MEN1':MEN1, 'MEN1n': MEN1n,
                'MEN2':MEN2, 'MEN2n': MEN2n, 'MEN3':MEN3, 'MEN3n': MEN3n, 'MEN4':MEN4, 'MEN4n': MEN4n, 'MEN5':MEN5,
                'MEN5n':MEN5n, 'HT1': HT1, 'HT1n': HT1n, 'HT2': HT2, 'HT2n': HT2n, 'HT3': HT3, 'HT3n': HT3n, 'HT4': HT4,
                'HT4n': HT4n, 'HT5': HT5, 'HT5n': HT5n, 'script5':script5, 'div5':div5}

    return render(request, 'index.html', contexto)

def añadir(request):
    id = request.GET.get('id')
    print(id)

    return render(request, 'index2.html')

def mongo_test():
    try:
        client = MongoClient()
        # Use twitterdb database. If it doesn't exist, it will be created.
        db = client['test']
        mycol = db['coltest']
        post = {"author": "Mike",
                "text": "My first blog post!",
                "tags": ["mongodb", "python", "pymongo"]}
        mycol.insert_one(post)

    except Exception as e:
        print(e)

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def leerTuits(id):
    if (id=='1'):
        df = pd.read_json('./datos/FinalLaVoz2016_2000.json', orient='columns')
        nombrePrograma = 'LA VOZ'
    elif (id=='2'):
        df = pd.read_json('./datos/Eurovision_2000.json', orient='columns')
        nombrePrograma = 'EUROVISIÓN'
    elif (id=='3'):
        df = pd.read_json('./datos/ElPrincipeFinal_2000.json', orient='columns')
        nombrePrograma = 'EL PRÍNCIPE'
    elif (id=='4'):
        df = pd.read_json('./datos/EmbajadaEH_2000.json', orient='columns')
        nombrePrograma = 'EMBAJADA'
    elif (id=='5'):
        df = pd.read_json('./datos/MDTE02S03_2000.json', orient='columns')
        nombrePrograma = 'EL MINISTERIO DEL TIEMPO'
    elif (id=='6'):
        df = pd.read_json('./datos/objetivoiglesias26J_2000.json', orient='columns')
        nombrePrograma = 'OBJETIVO IGLESIAS'
    else:
        df = pd.read_json('./datos/FinalLaVoz2016_2000.json', orient='columns')
        nombrePrograma = 'LA VOZ'

    return df, nombrePrograma

def pieChart(df):
    android = 0
    iphone = 0
    otros = 0
    for source in df['source']:
        if (source == '<a href="http://twitter.com/download/android" rel="nofollow">Twitter for Android</a>'):
            android = android + 1
        elif (source == '<a href="http://twitter.com/download/iphone" rel="nofollow">Twitter for iPhone</a>'):
            iphone = iphone + 1
        else:
            otros = otros + 1

    total = android + iphone + otros

    androidEJE = android / total
    iphoneEJE = (iphone / total) + androidEJE

    iphonePERCENT = iphone / total * 100
    otrosPERCENT = otros / total * 100

    iphonePERCENTfloat = round(iphonePERCENT, 0)
    otrosPERCENTfloat = round(otrosPERCENT, 0)
    androidPERCENTfloat = 100 - otrosPERCENTfloat - iphonePERCENTfloat

    androidPERCENTstring = "%.0f" % androidPERCENTfloat
    iphonePERCENTstring = "%.0f" % iphonePERCENTfloat
    otrosPERCENTstring = "%.0f" % otrosPERCENTfloat

#
    # define starts/ends for wedges from percentages of a circle
    sizes = [0, androidEJE, iphoneEJE, 1]
    starts = [p * 2 * pi for p in sizes[:-1]]
    ends = [p * 2 * pi for p in sizes[1:]]
    #legend = ["Android", "Iphone", "Otros"]


    # a color for each pie piece
    colors = ["lightcoral", "yellowgreen", "lightskyblue"]

    p = figure(x_range=(-1, 1), y_range=(-1, 1), plot_height=250, plot_width=350)

    p.wedge(x=0, y=0, radius=0.7, start_angle=starts, end_angle=ends, color=colors, legend=("Android " + androidPERCENTstring + "%"))
    p.wedge(x=0, y=0, radius=0.7, start_angle=starts[1], end_angle=ends[1], color=colors[1], legend=("Iphone " + iphonePERCENTstring + "%"))
    p.wedge(x=0, y=0, radius=0.7, start_angle=starts[2], end_angle=ends[2], color=colors[2], legend=("Otros " + otrosPERCENTstring + "%"))

    # display/save everythin
    output_file("pie.html")
    return(components(p))

def contarPalabras(df):
    tweet = ''
    for w in df.index:
        tweet += df.text[w]

    cache_english_stopwords = stopwords.words('english')
    cache_spanish_stopwords = stopwords.words('spanish')

    tweet_no_special_entities = re.sub(r'\&\w*;', '', tweet)
    tweet_no_tickers = re.sub(r'\$\w*', '', tweet_no_special_entities)
    tweet_no_hyperlinks = re.sub(r'https?:\/\/.*\/\w*', '', tweet_no_tickers)
    tweet_no_hashtags = re.sub(r'#\w*', '', tweet_no_hyperlinks)
    tweet_no_mentions = re.sub(r'@\w*', '', tweet_no_hashtags)
    tweet_no_numbers = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', '', tweet_no_mentions)
    tweet_no_exclamaciones = re.sub(r'¡\w*', '', tweet_no_numbers)
    tweet_no_interrogaciones = re.sub(r'¿\w*', '', tweet_no_exclamaciones)
    tweet_no_simbolito = re.sub(r'«*', '', tweet_no_interrogaciones)
    tweet_no_simbolito2 = re.sub(r'»*', '', tweet_no_simbolito)
    tweet_no_punctuation = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet_no_simbolito2)
    tweet_no_https = re.sub(r'https', '', tweet_no_punctuation)
    tweet_no_http = re.sub(r'http', '', tweet_no_https)
    tweet_no_small_words = re.sub(r'\b\w{1,3}\b', '', tweet_no_http)
    tweet_no_whitespace = re.sub(r'\s\s+', ' ', tweet_no_small_words)
    tweet_no_whitespace = tweet_no_whitespace.lstrip(' ')  # Remove single space remaining at the front of the tweet.
    tweet_no_emojis = ''.join(c for c in tweet_no_whitespace if
                              c <= '\uFFFF')  # Apart from emojis (plane 1), this also removes historic scripts and mathematical alphanumerics (also plane 1), ideographs (plane 2) and more.
    # print('No emojis:', tweet_no_emojis, '\n')
    # Tokenize: Change to lowercase, reduce length and remove handles
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=True,
                           strip_handles=True)  # reduce_len changes, for example, waaaaaayyyy to waaayyy.
    tw_list = tknzr.tokenize(tweet_no_emojis)
    # print('Tweet tokenize:', tw_list, '\n')
    # Remove stopwords
    list_no_stopwords = [i for i in tw_list if i not in cache_english_stopwords]
    # print('No stop words:', list_no_stopwords, '\n')
    list_no_stopwords = [i for i in list_no_stopwords if i not in cache_spanish_stopwords]
    # print('No stop words:', list_no_stopwords, '\n')
    # Final filtered tweet
    tweet_filtered = ' '.join(list_no_stopwords)
    # print('Final tweet:', tweet_filtered)

    lista_palabras = tweet_filtered.split()

    frecuenciaPalab = []
    for w in lista_palabras:
        frecuenciaPalab.append(lista_palabras.count(w))

    df_test = pd.DataFrame({'x': lista_palabras, 'y': frecuenciaPalab})

    df_simplyfied = df_test.drop_duplicates(keep='first')

    df_ordered = df_simplyfied.sort_values('y', ascending=False)

    results = df_ordered.head(11)

    results2 = results.tail(10)

    m = results2.as_matrix()

    # Set the x_range to the list of categories above
    p = figure(x_range=m[:, 0], plot_height=250, plot_width=350)

    # Categorical values can also be used as coordinates
    p.vbar(x=m[:, 0], top=m[:, 1], width=0.9, color='lightcoral')
    # Set some properties to make the plot look better
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = 45
    p.y_range.start = 0

    return(components(p))


def topRT(df):
    tweet_df = df.sort_values(by='retweet_count', ascending=False)
    tweet_df = tweet_df.reset_index(drop=True)
    return tweet_df['retweet_count'].iloc[0]

def topFAV(df):
    tweet_df = df.sort_values(by='favorite_count', ascending=False)
    tweet_df = tweet_df.reset_index(drop=True)
    return tweet_df['favorite_count'].iloc[0]


def topHT(df):
    tag_dict = {}

    for i in df.index:
        tweet_text = df.iloc[i]['text']
        tweet = tweet_text.lower()
        tweet_tokenized = tweet.split()

        for word in tweet_tokenized:
            if (word[0:1] == '#' and len(word) > 1):
                key = word
                if key in tag_dict:
                    tag_dict[key] += 1
                else:
                    tag_dict[key] = 1

    top_tags = dict(sorted(tag_dict.items(), key=operator.itemgetter(1), reverse=True)[:10])
    top_tags_sorted = sorted(top_tags.items(), key=lambda x: x[1])[::-1]
    return top_tags_sorted[1][0]


def topMEN(df):
    mention_dict = {}

    for i in df.index:
        tweet_text = df.iloc[i]['text']
        tweet = tweet_text.lower()
        tweet_tokenized = tweet.split()

        for word in tweet_tokenized:

            if (word[0:1] == '@' and len(word) > 1):
                key = word
                if key in mention_dict:
                    mention_dict[key] += 1
                else:
                    mention_dict[key] = 1

    top_mentions = dict(sorted(mention_dict.items(), key=operator.itemgetter(1), reverse=True)[:10])
    top_mentions_sorted = sorted(top_mentions.items(), key=lambda x: x[1])[::-1]
    for mention in top_mentions_sorted:
        return mention[0]

def allHT(df):
    tag_dict = {}

    for i in df.index:
        tweet_text = df.iloc[i]['text']
        tweet = tweet_text.lower()
        tweet_tokenized = tweet.split()

        for word in tweet_tokenized:
            if (word[0:1] == '#' and len(word) > 1):
                key = word
                if key in tag_dict:
                    tag_dict[key] += 1
                else:
                    tag_dict[key] = 1

    top_tags = dict(sorted(tag_dict.items(), key=operator.itemgetter(1), reverse=True)[:10])
    top_tags_sorted = sorted(top_tags.items(), key=lambda x: x[1])[::-1]
    HT1 = top_tags_sorted[1][0]
    HT1n = str(top_tags_sorted[1][1])
    HT2 = top_tags_sorted[2][0]
    HT2n = str(top_tags_sorted[2][1])
    HT3 = top_tags_sorted[3][0]
    HT3n = str(top_tags_sorted[3][1])
    HT4 = top_tags_sorted[4][0]
    HT4n = str(top_tags_sorted[4][1])
    HT5 = top_tags_sorted[5][0]
    HT5n = str(top_tags_sorted[5][1])
    return HT1, HT1n, HT2, HT2n, HT3, HT3n, HT4, HT4n, HT5, HT5n

def allMEN(df):
    mention_dict = {}

    for i in df.index:
        tweet_text = df.iloc[i]['text']
        tweet = tweet_text.lower()
        tweet_tokenized = tweet.split()

        for word in tweet_tokenized:

            if (word[0:1] == '@' and len(word) > 1):
                key = word
                if key in mention_dict:
                    mention_dict[key] += 1
                else:
                    mention_dict[key] = 1

    top_mentions = dict(sorted(mention_dict.items(), key=operator.itemgetter(1), reverse=True)[:10])
    top_mentions_sorted = sorted(top_mentions.items(), key=lambda x: x[1])[::-1]
    MEN1 = top_mentions_sorted[0][0]
    MEN1n = str(top_mentions_sorted[0][1])
    MEN2 = top_mentions_sorted[1][0]
    MEN2n = str(top_mentions_sorted[1][1])
    MEN3 = top_mentions_sorted[2][0]
    MEN3n = str(top_mentions_sorted[2][1])
    MEN4 = top_mentions_sorted[3][0]
    MEN4n = str(top_mentions_sorted[3][1])
    MEN5 = top_mentions_sorted[4][0]
    MEN5n = str(top_mentions_sorted[4][1])
    return MEN1, MEN1n, MEN2, MEN2n, MEN3, MEN3n, MEN4, MEN4n, MEN5, MEN5n
def allFAV(df):
    tweet_df = df.sort_values(by='favorite_count', ascending=False)
    tweet_df = tweet_df.reset_index(drop=True)
    FAV1t = tweet_df['text'].iloc[0]
    FAV1n = tweet_df['favorite_count'].iloc[0]
    FAV2t = tweet_df['text'].iloc[1]
    FAV2n = tweet_df['favorite_count'].iloc[1]
    FAV3t = tweet_df['text'].iloc[2]
    FAV3n = tweet_df['favorite_count'].iloc[2]
    FAV4t = tweet_df['text'].iloc[3]
    FAV4n = tweet_df['favorite_count'].iloc[3]
    FAV5t = tweet_df['text'].iloc[4]
    FAV5n = tweet_df['favorite_count'].iloc[4]
    FAV6t = tweet_df['text'].iloc[5]
    FAV6n = tweet_df['favorite_count'].iloc[5]
    FAV7t = tweet_df['text'].iloc[6]
    FAV7n = tweet_df['favorite_count'].iloc[6]
    FAV8t = tweet_df['text'].iloc[7]
    FAV8n = tweet_df['favorite_count'].iloc[7]
    FAV9t = tweet_df['text'].iloc[8]
    FAV9n = tweet_df['favorite_count'].iloc[8]
    FAV10t = tweet_df['text'].iloc[9]
    FAV10n = tweet_df['favorite_count'].iloc[9]
    return FAV1t, FAV1n, FAV2t, FAV2n, FAV3t, FAV3n, FAV4t, FAV4n, FAV5t, FAV5n, FAV6t, FAV6n, FAV7t, FAV7n, FAV8t, FAV8n, FAV9t, FAV9n, FAV10t, FAV10n

def allRT(df):
    tweet_df = df.sort_values(by='retweet_count', ascending=False)
    tweet_df = tweet_df.reset_index(drop=True)
    RT1t = tweet_df['text'].iloc[0]
    RT1n = tweet_df['retweet_count'].iloc[0]
    RT2t = tweet_df['text'].iloc[1]
    RT2n = tweet_df['retweet_count'].iloc[1]
    RT3t = tweet_df['text'].iloc[2]
    RT3n = tweet_df['retweet_count'].iloc[2]
    RT4t = tweet_df['text'].iloc[3]
    RT4n = tweet_df['retweet_count'].iloc[3]
    RT5t = tweet_df['text'].iloc[4]
    RT5n = tweet_df['retweet_count'].iloc[4]
    RT6t = tweet_df['text'].iloc[5]
    RT6n = tweet_df['retweet_count'].iloc[5]
    RT7t = tweet_df['text'].iloc[6]
    RT7n = tweet_df['retweet_count'].iloc[6]
    RT8t = tweet_df['text'].iloc[7]
    RT8n = tweet_df['retweet_count'].iloc[7]
    RT9t = tweet_df['text'].iloc[8]
    RT9n = tweet_df['retweet_count'].iloc[8]
    RT10t = tweet_df['text'].iloc[9]
    RT10n = tweet_df['retweet_count'].iloc[9]
    return RT1t, RT1n, RT2t, RT2n, RT3t, RT3n, RT4t, RT4n, RT5t, RT5n, RT6t, RT6n, RT7t, RT7n, RT8t, RT8n, RT9t, RT9n, RT10t, RT10n

def tuitsPorHora(df):
    # Time-series impressions (DOW, HOD, etc) (0 = Sunday... 6 = Saturday)
    gmt_offset = +1

    # Create proper datetime column, apply local GMT offset
    df['ts'] = pd.to_datetime(df['created_at'])
    df['ts'] = df.ts + pd.to_timedelta(gmt_offset, unit='h')

    # Add hour of day column
    df['hod'] = [t.hour for t in df.ts]

    hod_count = {}
    # Process tweets, collect stats
    for i in df.index:
        hod = df.iloc[i]['hod']

        if hod in hod_count:
            hod_count[hod] += 1
        else:
            hod_count[hod] = 1


    hod_count2 = collections.OrderedDict(sorted(hod_count.items()))

    x = hod_count2.keys()
    y = hod_count2.values()

    x_list = list(x)
    y_list = list(y)

    df_test = pd.DataFrame({'x': x_list, 'y': y_list})

    df_ordered = df_test.sort_values('x', ascending=True)

    m = df_ordered.as_matrix()

    test_x = list(map(str, m[:, 0]))
    test_y = list(map(str, m[:, 1]))

    p2 = figure(x_range=test_x, plot_height=250, plot_width=350)

    # Categorical values can also be used as coordinates
    p2.vbar(x=test_x, top=test_y, width=0.9, color='yellowgreen')
    # Set some properties to make the plot look better
    p2.xgrid.grid_line_color = None
    p2.y_range.start = 0

    return (components(p2))

def tuitsPorDia(df):
    # Time-series impressions (DOW, HOD, etc) (0 = Sunday... 6 = Saturday)
    gmt_offset = +1

    # Create proper datetime column, apply local GMT offset
    df['ts'] = pd.to_datetime(df['created_at'])
    df['ts'] = df.ts + pd.to_timedelta(gmt_offset, unit='h')

    # Add day of week column
    df['dow'] = [t.dayofweek for t in df.ts]
    dow_count = {}
    # Process tweets, collect stats
    for i in df.index:
        dow = df.iloc[i]['dow']

        if dow in dow_count:
            dow_count[dow] += 1
        else:
            dow_count[dow] = 1


    dow_count2 = collections.OrderedDict(sorted(dow_count.items()))

    x = dow_count2.keys()
    y = dow_count2.values()

    x_list = list(x)
    y_list = list(y)



    df_test = pd.DataFrame({'x': x_list, 'y': y_list})

    df_ordered = df_test.sort_values('x', ascending=True)

    vals_to_replace = {0: 'Lun', 1: 'Mar', 2: 'Mie', 3: 'Jue', 4: 'Vie', 5: 'Sab', 6: 'Dom'}
    df_ordered['x'] = df_ordered['x'].map(vals_to_replace)

    m = df_ordered.as_matrix()

    test_x = list(map(str, m[:, 0]))
    test_y = list(map(str, m[:, 1]))

    p3 = figure(x_range=test_x, plot_height=250, plot_width=350)

    # Categorical values can also be used as c#oordinates
    p3.vbar(x=test_x, top=test_y, width=0.9, color='lightskyblue')
    # Set some properties to make the plot look better
    p3.xgrid.grid_line_color = None
    #p3.xaxis.major_label_orientation = 45
    p3.y_range.start = 0


    return (components(p3))

def Polaridad(df):
    neutros = 0
    positivos = 0
    negativos = 0
    for w in df.index:
        analysis = TextBlob(clean_tweet(df.text[w]))
        if analysis.sentiment.polarity > 0:
            positivos += 1
        elif analysis.sentiment.polarity < 0:
            negativos += 1
        else:
            neutros += 1

    total = positivos + negativos + neutros

    positivosEJE = positivos / total
    negativosEJE = (negativos / total) + positivosEJE

    negativosPERCENT = negativos / total * 100
    neutrosPERCENT = neutros / total * 100

    negativosPERCENTfloat = round(negativosPERCENT, 0)
    neutrosPERCENTfloat = round(neutrosPERCENT, 0)
    positivosPERCENTfloat = 100 - neutrosPERCENTfloat - negativosPERCENTfloat

    positivosPERCENTstring = "%.0f" % positivosPERCENTfloat
    negativosPERCENTstring = "%.0f" % negativosPERCENTfloat
    neutrosPERCENTstring = "%.0f" % neutrosPERCENTfloat

    #
    # define starts/ends for wedges from percentages of a circle
    sizes = [0, positivosEJE, negativosEJE, 1]
    starts = [p * 2 * pi for p in sizes[:-1]]
    ends = [p * 2 * pi for p in sizes[1:]]

    # a color for each pie piece
    colors = ["yellowgreen", "lightcoral",  "lightskyblue"]

    p = figure(x_range=(-1, 1), y_range=(-1, 1), plot_height=250, plot_width=350)

    p.wedge(x=0, y=0, radius=0.7, start_angle=starts, end_angle=ends, color=colors,
            legend=("Positivos " + positivosPERCENTstring + "%"))
    p.wedge(x=0, y=0, radius=0.7, start_angle=starts[1], end_angle=ends[1], color=colors[1],
            legend=("Negativos " + negativosPERCENTstring + "%"))
    p.wedge(x=0, y=0, radius=0.7, start_angle=starts[2], end_angle=ends[2], color=colors[2],
            legend=("Neutros " + neutrosPERCENTstring + "%"))


    return(components(p))