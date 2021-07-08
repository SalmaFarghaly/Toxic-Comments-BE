# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
from comment import comment
import os
import uvicorn
from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from keras.preprocessing.sequence import pad_sequences
from cleaning import cleancomment
import string
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

from attention import attention
from padding import custom_padding
from Bert import classifier_model

# 2. Create the app object
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#load tokenizer

TfIDFtokenizer= pickle.load(open("Models/TfidfNaiveClassifier (1).pkl", 'rb'))
NB= pickle.load(open("Models/NaiveClassifier2.pkl", 'rb'))

GRUmodel = load_model("Models/deep-learning-model-full-v0.03.01.weights-improvement-01-0.9244.hdf5",custom_objects={'attention': attention})
tokenizer= pickle.load(open("Models/tokenizer.pickle", 'rb'))

RFmodel=pickle.load(open("Models/xgb_Final3.pkl", 'rb'))


cwd = os.getcwd()
EMBEDDING_FILE_FASTTEXT=cwd+"\crawl-300d-2M.vec"
EMBEDDING_FILE_TWITTER=cwd+"\glove.twitter.27B.200d.txt"
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index_ft = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE_FASTTEXT,encoding='utf-8'))
embeddings_index_tw = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE_TWITTER,encoding='utf-8'))



cols_features = ['count_word','capitals', 'count_unique_word',  'count_punctuations','word_unique_percent' ,'punct_percent'
                 , 'num_exclamation_marks', 'num_question_marks', 'you_count','mentions','smilies' ,'symbols']

def get_features(df):
    
    df['count_word']=df["tweet"].apply(lambda comment: len(str(comment).split()))
    df['capitals'] = df["tweet"].apply(lambda comment: sum(1 for c in str(comment) if c.isupper()))
    df['count_unique_word']=df["tweet"].apply(lambda comment: len(set(str(comment).split())))
    df["count_punctuations"] =df["tweet"].apply(lambda comment: len([c for c in str(comment) if c in  string.punctuation]))
    df['word_unique_percent']=df['count_unique_word']*100/df['count_word']
    df['punct_percent']=df['count_punctuations']*100/df['count_word']
    df['num_exclamation_marks'] = df['tweet'].apply(lambda comment: str(comment).count('!'))
    df['num_question_marks'] = df['tweet'].apply(lambda comment:  str(comment).count('?'))
    df['you_count'] = df['tweet'].apply(lambda comment: sum( str(comment).count(w) for w in ('you', 'You', 'YOU')))
    df['mentions'] = df['tweet'].apply(lambda comment: str(comment).count("@"))
    df['smilies'] = df['tweet'].apply(lambda comment: sum(str(comment).count(w) for w in (':-)', ':)', ';-)', ';)')))
    df['symbols'] = df['tweet'].apply(lambda comment: sum(str(comment).count(w) for w in '*&#$%“”¨«»®´·º½¾¿¡§£₤‘’'))


    scaler = MinMaxScaler().fit(df[cols_features])
    df[cols_features] = scaler.transform(df[cols_features])
    return df
# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Krish Youtube Channel': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_tweets(data:comment):

    print(data)
    data = data.dict()
    text=data['tweets']
    print(text)
    if len(text)==0:
        return {
            'prediction':[]
        }
    ans=cleancomment(text)

    # Naive Bayes prediction
    tweets_array = ans["lemmatize_text"]
    tweets = np.array(tweets_array.iloc[0:])
    Train_X_Tfidf = TfIDFtokenizer.transform(tweets)
    NBpred = NB.predict(Train_X_Tfidf )

    # predictions_NB*=100
    NBpred=np.around(NBpred)
    print(NBpred)
    # pred_valuesNB=[1 if i >50  else 0 for i in predictions_NB ]

    # GRU prediction
    test_df=get_features(ans)
    testX = test_df['lemmatize_text'].astype(str)
    testFeat = test_df[cols_features]
    testX_sequence = tokenizer.texts_to_sequences(testX)
    padTesting=custom_padding(testX_sequence)
    testX_pad  = pad_sequences(padTesting)
    GRUpred = GRUmodel.predict([testX_pad,testFeat],batch_size = 32,verbose=1)

    # pred_values*=100
    # pred_valuesGRU=[1 if i >50  else 0 for i in pred_values ]
    GRUpred=tf.reshape(GRUpred, [-1]).numpy()
    GRUpred=np.around(GRUpred)

   # Get BERT predictions
    print("GRUPREDSSS",GRUpred)
    # print(ans['lemmatize_text'])
    original_results = tf.sigmoid(classifier_model(tf.constant(ans['lemmatize_text'])))
    print(original_results)
    BertPred=tf.reshape(original_results, [-1]).numpy()
    BertPred=np.around(BertPred)
    print(BertPred)
    # float_arr = np.vstack(original_results[:, :]).astype(np.float)
    # float_arr.flatten()
    # pred_values*=100
    # pred_valuesBERT=[1 if i >50  else 0 for i in pred_values ]
    # pred_valuesBERT=np.around(original_results.flatten())
    predNBlist=NBpred.tolist()
    predGRUlist=GRUpred.tolist()
    predBertlist=BertPred.tolist()

    # Get RF predictions
    NBdf = pd.DataFrame(NBpred)
    GRUdf = pd.DataFrame(GRUpred)
    Bertdf = pd.DataFrame(BertPred)

    concat = np.concatenate((testX_pad,Bertdf,NBdf,GRUdf), axis = 1)
    FinalPred=RFmodel.predict(concat)
    print("FF  ",FinalPred)
    FinalPredlist=FinalPred.tolist()

    return {
        # 'predNB': predNBlist,
        # 'predGRU':predGRUlist,
        # 'predBERT':predBertlist,
        'FinalPred':FinalPredlist
    }



# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload