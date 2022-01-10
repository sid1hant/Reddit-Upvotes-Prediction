import matplotlib as plt
import pandas as pd
import numpy as np
import os
import praw
import pickle
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from textblob import TextBlob
import re
from sklearn.preprocessing import OneHotEncoder
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import xgboost as xgb
from xgboost import XGBRegressor

def Subjectivity(text):
    return TextBlob(text).sentiment.subjectivity
def Polarity(text):
    return TextBlob(text).sentiment.polarity
def word_count(text):
    wordList = re.sub("[^\w]", " ", text).split()
    return len(wordList)
def clean_message(text):
    text = re.sub(r'[^\w\s]', '', text)
    l_text = " ".join(word for word in text.lower().split() if word not in ENGLISH_STOP_WORDS)
    return l_text
def spaceURLs(df, column):
    data = df.copy()
    data.loc[:, column] = data[column].str.replace('\[|\]', ' ', regex=True)
    return data


#Replace URLs

def replaceURL(df, column):
    data = df.copy()
    data.loc[:, column] = data[column].str.replace('\(http\S+', 'URL', regex=True)
    return data


#Removing symbols

def removeSymbols(df, column):
    data = df.copy()
    data.loc[:, column] = data[column].str.replace('/r/', '', regex=True)
    data.loc[:, column] = data[column].str.replace('[^\.\'A-Za-z0-9]+', ' ', regex=True)
    return data


#Removing numbers

def removeNumbers(df, column):
    data = df.copy()
    data.loc[:, column] = data[column].str.replace('\S*\d\S*', '', regex=True)
    return data

with open('senti.pkl', "rb") as u:
    senti = pickle.load(u)


# Captures the path of current folder
#curr_path = os.path.dirname(os.path.realpath(__file__))

with open('encoding.pkl', "rb") as b:
    enc = pickle.load(b)

model_xgb_2 = xgb.Booster()
model_xgb_2.load_model("test_model2.bin")

# with open('model_dec.pkl', "rb") as a:
#     model12 = pickle.load(a)



#x=app.text

def post_new(x):
  text1=str(x)
  reddit = praw.Reddit(client_id='FSH5ERlb1UGg4MY46x6lNQ', client_secret='9jLcpqCc3z_Fh-6jqEXgrdqUwQemgA', user_agent='new')
  submission = reddit.submission(url=text1)
  posts = []
  posts.append([submission.title,submission.is_self,submission.num_comments,submission.over_18,submission.gilded])
  posts = pd.DataFrame(posts,columns=['title', 'is_self',
       'num_comments', 'over_18', 'gilded'])
  posts['title'] = posts.title.astype('category')
  posts['subjectivity'] = posts['title'].str.count('I ')
  posts['no_quoted'] = posts.title.str.count('&gt;')
  posts['character_count'] = posts['title'].map(lambda x : len(x))
  posts['word_count'] = posts['title'].map(lambda x : word_count(x))
  posts['character_count'] = posts.character_count.astype('int')
  posts['word_count'] = posts.word_count.astype('int')
  data = spaceURLs(posts, 'title')
  data = replaceURL(posts, 'title')
  data = removeNumbers(posts, 'title')
  data = removeSymbols(posts, 'title')
  data['Polarity'] = data['title'].apply(Polarity)
  data['rating'] = data['title'].apply(senti.polarity_scores)
  reddit3 = pd.concat([data.drop(['rating'], axis=1), data['rating'].apply(pd.Series)], axis=1)
  reddit3.drop(["title"],axis=1, inplace=True)
  categories = ['over_18']
  test_encoded = enc.transform(reddit3[categories])
  col_names = [False, True]
  test_ohe = pd.DataFrame(test_encoded.todense(), columns=col_names)
  categories1 = ['is_self']
  test_encoded1= enc.transform(reddit3[categories1])
  col_names1 = [False, True]
  test_ohe1 = pd.DataFrame(test_encoded1.todense(), columns=col_names1)
  X = pd.concat([reddit3,test_ohe,test_ohe1], axis=1)
  X.drop(["is_self", "over_18"], axis=1, inplace=True)
  X= X.rename(columns={X.columns[13]: 'False'})
  X = X.rename(columns={X.columns[14]: 'True'})
  cols = []
  count = 1
  for column in X.columns:
    if column == 'False':
        cols.append(f'False_{count}')
        count+=1
        continue
    cols.append(column)
  X.columns = cols
  
  cols = []
  count = 1
  for column in X.columns:
    if column == 'True':
        cols.append(f'True_{count}')
        count+=1
        continue
    cols.append(column)
  X.columns = cols
  dtest = xgb.DMatrix(X)
  score = model_xgb_2.predict(dtest)
  output=round(score[0],2)
  return ("Predicted Score is: {}".format(output))

# def predications(posts):
#     print("Duration predicted")

#     return int(pred)
