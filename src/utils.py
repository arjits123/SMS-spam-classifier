import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging

import numpy as np
import pandas as pd
import dill

import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import string
punctuations = string.punctuation

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from sklearn.metrics import accuracy_score,confusion_matrix, precision_score


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e,sys)

def transform_text(text):
    # Convert into lower case
    text = text.lower()
    #tokenisation
    text = nltk.word_tokenize(text=text)
    
    #Remove special characters
    y = []
    for word in text:
        if word.isalnum():
            y.append(word)
    
    text = y[:]
    y.clear()

    # Remove stop words and punctuations
    for i in text:
        if i not in stop_words and i not in punctuations:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)