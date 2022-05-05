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
from tensorflow.keras import layers


##########
##### CONFIG
##########

st.set_page_config(page_title="SMS analysis",
                   page_icon="📈",
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

st.sidebar.title("Table des matières")
week = st.sidebar.radio(
    'Etapes de présentation :',
    ('Accueil', 'NLP et preprocessing', 'Présentation des modèles', 'Invite de saisie de SMS')
    )


##########
##### Set up main app.
##########