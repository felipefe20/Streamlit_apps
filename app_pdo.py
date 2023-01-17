import streamlit as st
import pandas as pd
import numpy as np
import glob
import re

#text processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from string import punctuation
import unicodedata

nltk.download('stopwords')
stemmer = SnowballStemmer('english')
nltk.download('punkt')

stop_words = set(stopwords.words('english')) 

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


def fast_preproc(text):
  #limpieza de columna v3_tags

  text = text.lower()
  text = ''.join(c for c in text if not c.isdigit())
  text = ''.join(c for c in text if c not in punctuation)
  text = remove_accents(text)
  words = word_tokenize(text)
  words = [stemmer.stem(word) for word in words]  
  words = [word for word in words if not word in stop_words] 
  try:
    text = " ".join(str(word) for word in words)
  except Exception as e:
    print(e)
    pass
  return text



#Modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report,confusion_matrix

#Predice para el conjunto de testeo.


import pickle

vec_model=pickle.load(open("vec_model.pkl","rb"))
classifier_loaded=pickle.load(open("model.pkl","rb"))


st.write("Import table to predict")

import streamlit as st

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df = pd.read_excel(uploaded_file)
  st.write(df)


#df=pd.read_excel(r"Raw Data\GF_VOD_TEL_APR_2.xlsx", sheet_name="Raw Data")

df = df.replace(to_replace=r'label_brs_v3_', value='', regex=True)
df = df.replace(to_replace=r'"', value='', regex=True)
df['v3_tags'] = df['v3_tags'].apply(lambda x: re.sub (r"[\[\]]", '', x, flags= re.IGNORECASE))
df = df.replace(to_replace=r',', value=' ', regex=True)
df = df.replace(to_replace=r'_', value=' ', regex=True)


def prediction_step(text):
    Xt_new = [fast_preproc(str(text))]
    trans_new_doc = vec_model.transform(Xt_new)
    #trans_new = vec.transform(trans_new_doc) #Use same TfIdfVectorizer
    #print("\nPredicted result: " + str(classifier_loaded.predict(trans_new_doc)))
    return str(classifier_loaded.predict(trans_new_doc))


df["Prediction"]=df["v3_tags"].apply(prediction_step)


st.write(df[["job_id","v3_tags","Prediction"]])

st.write("Download button")

to_download=df[["job_id","v3_tags","Prediction"]]

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(to_download)
st.download_button(
    label="Download data as csv",
    data=csv,
    file_name='prediction.csv',
    mime='text/csv'
)
