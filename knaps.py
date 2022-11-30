import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC

st.write(""" 
# Cek data
""")

st.write("=========================================================================")

tab1, tab2, tab3, tab4 = st.tabs(["Import Data", "Preprocessing", "Modelling", "Evalutions"])

with tab1:
    st.write("Import Data")
    wine = pd.read_csv("https://raw.githubusercontent.com/elmatiaaa/Machine-Learning/main/winequality-red.csv")
    st.dataframe(wine)

with tab2:
    quality_series = wine.loc[:, "quality"]
    quality_categorical_series = pd.cut(quality_series, [0, 5, 10], labels=["bad", "good"])
    wine["quality"] = quality_categorical_series
    wine
    



   
