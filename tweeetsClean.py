## Import libraries

import pandas as pd
import re
import string
import numpy as np
from tqdm import tqdm
import random

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
stop=set(stopwords.words('english'))

## Loading the data 

#tweet= pd.read_csv('data/covdsamp1000K.csv',delimiter=",", low_memory=False)
tweet= pd.read_csv('data/covdsamp100.csv',delimiter=",", low_memory=False)

tw = tweet[["id", "text"]] # retains two columns out of 37, the text columne is being cleaned  

## remove urls

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

tw.text=tw.text.apply(lambda x : remove_URL(x))

## Removing HTML Tags

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

tw.text=tw.text.apply(lambda x : remove_html(x))

## Remove emoji

# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    emoji_pattern = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

tw.text=tw.text.apply(lambda x: remove_emoji(x))

## Removing punctuations

def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

tw.text=tw.text.apply(lambda x : remove_punct(x))

tw.to_csv("tw_out.csv", index=False)
