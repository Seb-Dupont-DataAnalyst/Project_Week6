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
import pandas as pd
import tensorflow
from tensorflow.keras import layers


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

df = load_df("https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/Project_Week6/main/dfsms_nlp.csv")


##########
##### Set up sidebar.
##########

st.sidebar.title("INDEX")
panelChoice = st.sidebar.radio('Presentation:', ('Home', 'The dataset', 'Preprocessing & NLP', 'The models', 'SMS analysis'))

##########
##### Set up main app.
##########

if (panelChoice == 'Home'):

    st.write('You are in the "HOME" panel.')

elif (panelChoice == 'The dataset'):

    st.write('The "SMS Spam Collection": ')
    st.write('- 425 SMS extracted from the Grumbletext Web site (UK forum).')
    st.write('- 450 SMS extracted from the Caroline Tag\'s PhD Theses.')
    st.write('- 3375 SMS extracted from the NUS SMS Corpus (NSC ; 10 000 legitimate messages from Singapore).')
    st.write('- 1002 SMS ham messages and 322 spam messages from the SMS Spam Corpus v.0.1 Big.')

elif (panelChoice == 'Preprocessing & NLP'):

    st.write('You are in the "Preprocessing & NLP" panel.')

elif (panelChoice == 'The models'):

    st.write('SCIKIT-LEARN')
    st.write('TENSORFLOW')

else: 
    
    smsToAnalyze = st.text_input('Type your message:')

    ### the radio buttons about the 2 different models will be displayed
    modelChoice = st.radio('Choose your model:', ('Scikit-Learn', 'TensorFlow'))

    startButton = st.button('Start the analysis')

    if startButton:
        
        if (smsToAnalyze != ''):
            
            st.write('Your message is " ', smsToAnalyze, '".')

            if (modelChoice == 'Scikit-Learn'):
                ### start the analysis with Scikit-Learn
                st.write('You chose the model " ', modelChoice, '".')
            else:
                ### start the analysis with TensorFlow
                st.write('You chose the model " ', modelChoice, '".')

        else:
            st.write('Please start to write a SMS before clicking on the button.')

    else:
        st.write('')