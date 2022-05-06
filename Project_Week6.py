##########
##### PROJECT: Project_Week6.py
##### AUTHORS: 
##### LAST UPDATE: 05/05/2022
##########

##########
##### IMPORTS
##########

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
import streamlit as st
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
import io
import re
import string
from tensorflow.keras import layers
import pickle
import spacy
from nltk.stem import SnowballStemmer
import nltk
nltk.download('popular')
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from nltk.corpus import stopwords
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
import requests
from io import BytesIO


##########
##### CONFIG
##########

st.set_page_config(page_title="SMS analysis",
                   page_icon="📲",
                #    page_icon="📈",
                   layout="wide",
                   initial_sidebar_state="expanded")

#st.markdown('<style>' + open('style.css').read() + '</style>', unsafe_allow_html=True)

st.markdown("""
    <style>
    .titre {
        font-size:16px;
        font-weight:normal;
        margin:10px;
    }
    .text {
        font-size:25px;
        font-weight:normal;
        color:gray;
    }
    .sub_value {
        font-size:50px;
        font-weight:bold;
        line-height:1;
    }
    .value {
        font-size:80px;
        font-weight:bold;
        line-height:1;
    }
    </style>
    """, unsafe_allow_html=True)

##########
##### FUNCTIONS
##########

def space(n):
    """
    :param n: number of spaces
    :return: print n spaces
    """
    for n in range(n):
        st.title(" ")

def load_df(url):
    return pd.read_csv(url)


##########
##### DATASET
##########

dfsms = load_df("https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Project_Week6/main/dfsms_nlp.csv")


mLink = 'https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Project_Week6/main/sklearn_mlp_model.pkl'
mfile = BytesIO(requests.get(mLink).content)
mlp_model = pickle.load(mfile)

mLink2 = 'https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Project_Week6/main/count_vect_only.pkl'
mfile2 = BytesIO(requests.get(mLink2).content)
count_vect_model = pickle.load(mfile2)

dfsms = dfsms.dropna()

X = dfsms['message']
y = dfsms['target']

X_vect = count_vect_model.transform(X)
X_array = X_vect.toarray()

X_train, X_test, y_train, y_test = train_test_split(
    X_array,
    y,
    test_size=0.3,
    random_state=0
    )


def create_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='sigmoid'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(1, activation='hard_sigmoid')
    ])

    model.compile(optimizer='adam',
                loss="BinaryCrossentropy",
                metrics=['accuracy'])

    return model

clf = KerasClassifier(create_model,verbose=2, epochs=10)

pipeline_tensorflow = Pipeline([
                    ("clf",  clf)
                ]).fit(X_train, y_train)
nlp = spacy.load("en_core_web_sm")
stopwordsenglish = nltk.corpus.stopwords.words("english")


def wordsCleaning(tmpSMS, tmpTokenizer, stopwordsENG):

    endArray = []
    
    tmpWords = tmpTokenizer.tokenize(tmpSMS)

    ### parsing the words of the given array
    for tmpSingleWord in tmpWords:
    
        ### checking if the word is a stop word or not
        if tmpSingleWord in stopwordsENG:
            pass
        else:
            endArray.append(tmpSingleWord)

    return endArray


def purge(s):
    s_finale=[x for x in s if x not in string.punctuation]
    s_finale= " ".join(s_finale)
    l_finale = [x for x in s_finale.split(" ") if x.lower() not in stopwords.words("english") and x!=" "]
    return l_finale


def lemma(texts):
    list_sentence = []
    for text in texts :
        sent_tokens = nlp(text)
        for token in sent_tokens:
            list_sentence.append(token.lemma_)
            text_clean = " ".join(list_sentence)
    return text_clean


def nlp_preprocess_pipeline(tmpText):
    tmpDf = pd.DataFrame(data=[tmpText], columns = ['text'])
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tmpDf['words'] = tmpDf['text'].apply(lambda w: wordsCleaning(w, tokenizer, stopwordsenglish))
    tmpDf['message'] = tmpDf["words"].apply(purge)
    tmpDf['message'] = tmpDf['message'].apply(lemma)
    return tmpDf.iloc[0,2]

#pickle_in = requests.get('https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Project_Week6/main/sklearn_mlp_model.pkl', 'rb')
#mlp_model = pickle.load(pickle_in) 


#pickle_count_vect = requests.get('https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Project_Week6/main/count_vect_only.pkl', 'rb')
#count_vect_model = pickle.load(pickle_count_vect) 





##########
##### Set up sidebar.
##########

st.sidebar.title("Welcome :open_hands:")
panelChoice = st.sidebar.radio('', ('Home', 'The dataset', 'Preprocessing & NLP', 'The models', 'SMS analysis', 'Conclusion'))

##########
##### Set up main app.
##########

if (panelChoice == 'Home'):

    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        st.write('')
    with col2:
        st.title('Modeling with Neural Networks')
    with col3:
        st.write('')

    space(1)
    st.image('https://www.sms77.io/wp-content/uploads/SMS-Spam-Header.jpg', width=1200)
    

elif (panelChoice == 'The dataset'):

    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        st.write('')
    with col2:
        st.title('Dataset Exploration')
    with col3:
        st.write('')

    space(1)
    st.subheader('The "SMS Spam Collection": 📲')
    st.write('- 425 SMS extracted from the Grumbletext Web site (UK forum).')
    st.write('- 450 SMS extracted from the Caroline Tag\'s PhD Theses.')
    st.write('- 3375 SMS extracted from the NUS SMS Corpus (NSC ; 10 000 legitimate messages from Singapore).')
    st.write('- 1002 SMS ham messages and 322 spam messages from the SMS Spam Corpus v.0.1 Big.')
    space(1)
    st.write('A total of 5,574 messages divided between 4,827 SMS legitimate messages (86.6%) and 747 (13.4%) spam messages.')
      
    space(1)
    
    cols = st.columns(2)

    with cols[0]:
        st.subheader('Ham SMS Wordcloud :')
        st.image('https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Project_Week6/main/ham_wordcloud.png')

    with cols[1]:
        st.subheader('Spam SMS Wordcloud :')
        st.image('https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Project_Week6/main/spam_wordcloud.png')

elif (panelChoice == 'Preprocessing & NLP'):

    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        st.write('')
    with col2:
        st.title('Preprocessing & NLP')
    with col3:
        st.write('')  
    
    space(2)
    
    st.subheader('1. Removing the english stop words')
    
    st.image('https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Project_Week6/main/Hammeuh.png')
    st.write('Stopwords : not useful for text analysis, but the most common words in the sms')
    space(1)
    st.subheader('2. Stemming or Lemmatization')
    st.write('Words like "text", "texted" and "texting" share the same root')
    st.image('https://devopedia.org/images/article/227/6785.1570815200.png')
    st.write('Lemmatization identifies them so they are not treated as different words within the analysis')
    space(1)
    st.subheader('3. Vectorizing')
    st.write('Machine learning models can\'t work on words, we need to transform them into numerical vectors')
    st.write('Tests showed better results using the "CountVectorizer" method')
    space(1)
    st.subheader('4. Creating a pipeline')
    st.write('These pre-processing steps are applied to any submitted SMS')

elif (panelChoice == 'The models'):
    modelChoice = st.sidebar.radio('Models:', ('Scikit-Learn', 'Tensorflow'))
    

    
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        st.write('')
    with col2:
        st.title('Models comparison')
        space(1)
    with col3:
        st.write('')
        
    if modelChoice == 'Scikit-Learn':

        cols = st.columns(2)

            
        with cols[0]:
            
            st.subheader('Perceptron :')
            st.write('Using Tfidf :')
            st.image('https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Project_Week6/main/disp_ppn_cvect.png')
            st.write('Using CountVectorizer:')
            st.image('https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Project_Week6/main/disp_ppn_tfidf.png')

        with cols[1]:
            st.subheader('Multi-Layer Perceptron :')
            st.write('Without tuning')
            st.image('https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Project_Week6/main/disp_mlp_countvect.png')
            st.write('With tuning')
            st.image('https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Project_Week6/main/disp_mlp_countvect_tuned.png')

    

    if modelChoice == 'Tensorflow':

        cols = st.columns(3)

        with cols[0]:
            st.write('Without tuning :')
            st.image('https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Project_Week6/main/disp_countvect_tensorflow.png')

        with cols[1]:
            st.write('Tuning method 1 :')
            st.image('https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Project_Week6/main/disp_countvect_tensorflow_1.png')

        with cols[2]:   
            st.write('Tuning method 2 :')
            st.image('https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Project_Week6/main/disp_countvect_tensorflow_2.png')



elif (panelChoice == 'SMS analysis'): 
     
    st.title('The "Spam Checker"')
    title = st.text_input('Type your message and press enter')
    result ="" 
    #st.write('SMS to classify : ', title)
    choix_techno = st.radio('What technology do you want to use ?', ('Scikit Learn', 'Tensorflow'))

    if title != '':
        title_vect = nlp_preprocess_pipeline(title)
        
        if choix_techno == 'Scikit Learn':            
            process = mlp_model.predict([title_vect])
            if st.button("Launch process"):
                result = process
               

                if result == 0: 
                    st.success('The message you just typed is not a Spam')
                    st.image('https://sagamer.co.za/wp-content/uploads/2019/10/Drake-Hotline-Bling-approve.jpg')
                if result == 1:
                    st.warning('The message you just typed is a Spam')
                    st.image('https://litreactor.com/sites/default/files/images/column/headers/drake_ignores_writing_advice.jpg')
   
    
        if choix_techno == 'Tensorflow':
            
            count_vect = count_vect_model
            title_vect_tf = count_vect.transform([title_vect])  
            title_array = title_vect_tf.toarray()    
            process = pipeline_tensorflow.predict(title_array)
            if st.button("Launch process"):
                result = process
                if result == 0:
                    st.success('The message you just typed is not a Spam')
                    st.image('https://sagamer.co.za/wp-content/uploads/2019/10/Drake-Hotline-Bling-approve.jpg')
                if result == 1:
                    st.warning('The message you just typed is a Spam')
                    st.image('https://litreactor.com/sites/default/files/images/column/headers/drake_ignores_writing_advice.jpg')

elif (panelChoice == 'Conclusion'):
    space(2)
    col1, col2, col3 = st.columns([1, 4, 1])

    with col1:
        st.write('')
    with col2:
        st.image("https://marketingonline.gratis/wp-content/uploads/2021/12/gif-thanks-for-your-attention.gif", width=800)
        
    with col3:
        st.write('')
    