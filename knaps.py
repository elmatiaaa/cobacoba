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
    
    pd.set_option('max_columns', None)
    pd.options.display.float_format = "{:.3f}".format
    wine.describe()
    
    from sklearn import preprocessing

def convert_categorical_to_dummy(columns, dataframe):
    """
    Converts categorical columns to binary, results in n-1 columns,
    where n is the number of possible category values.
    """
    le = preprocessing.LabelEncoder()

    for col in columns:
        n = len(dataframe[col].unique())
        if n > 2:
            X = pd.get_dummies(dataframe[col])
            X = X.drop(X.columns[0], axis=1)
            dataframe[X.columns] = X
            dataframe.drop(col, axis=1, inplace=True)  # drop the original categorical variable (optional)
        else:
            le.fit(dataframe[col])
            dataframe[col] = le.transform(dataframe[col])
pre_dummy_column = wine["quality"].copy()
pre_dummy_column = pre_dummy_column.rename("quality (pre dummy)")

categorical_columns = wine.select_dtypes(['category']).columns
convert_categorical_to_dummy(categorical_columns, wine)

post_dummy_column = wine["quality"].copy()
post_dummy_column = post_dummy_column.rename("quality (post dummy)")

pd.concat([pre_dummy_column, post_dummy_column], axis=1)

   

   
