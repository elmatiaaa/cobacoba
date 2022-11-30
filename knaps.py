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

# Split Data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

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
    

with tab3:
    # pisahkan fitur dan label
    knn,naivebayes,decisiontree= st.tabs(
        ["K-Nearest Neighbor","naivebayes","decisiontree"])
    with knn:
      from sklearn.neighbors import KNeighborsClassifier
      knn = KNeighborsClassifier(n_neighbors=3)
      knn.fit(x_train,y_train)
      y_pred_knn = knn.predict(x_test) 
      accuracy_knn=round(accuracy_score(y_test,y_pred_knn)* 100, 2)
      acc_knn = round(knn.score(x_train, y_train) * 100, 2)
      label_knn = pd.DataFrame(
      data={'Label Test': y_test, 'Label Predict': y_pred_knn}).reset_index()
      st.success(f'Tingkat akurasi = {acc_knn}')
      st.dataframe(label_knn)

    with naivebayes:
        #Metrics
        from sklearn.metrics import make_scorer, accuracy_score,precision_score
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

        #Model Select
        from sklearn.model_selection import KFold,train_test_split,cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import  LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn import linear_model
        from sklearn.linear_model import SGDClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC, LinearSVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.preprocessing import LabelEncoder 
        le = LabelEncoder()
        y = le.fit_transform(y)
        gaussian = GaussianNB()
        gaussian.fit(x_train, y_train)
        y_pred = gaussian.predict(x_test) 
        accuracy_nb=round(accuracy_score(y_test,y_pred)* 100, 2)
        acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)

        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test,y_pred)
        precision =precision_score(y_test, y_pred,average='micro')
        recall =  recall_score(y_test, y_pred,average='micro')
        f1 = f1_score(y_test,y_pred,average='micro')
        print('Confusion matrix for Naive Bayes\n',cm)
        print('accuracy_Naive Bayes: %.3f' %accuracy)
        print('precision_Naive Bayes: %.3f' %precision)
        print('recall_Naive Bayes: %.3f' %recall)
        print('f1-score_Naive Bayes : %.3f' %f1)
        st.success(accuracy)
        label_nb = pd.DataFrame(
        data={'Label Test': y_test, 'Label Predict': y_pred})
        label_nb


    with decisiontree:
        from sklearn.tree import DecisionTreeClassifier
        d3 = DecisionTreeClassifier()
        d3.fit(x_train, y_train)
        y_predic = d3.predict(x_test)
        data_predic = pd.concat([pd.DataFrame(y_test).reset_index(drop=True), pd.DataFrame(y_predic, columns=["Predict"]).reset_index(drop=True)], axis=1)        
        from sklearn.metrics import accuracy_score
        a=f'acuraty = {"{:,.2f}".format(accuracy_score(y_test, y_predic)*100)}%'
        st.success(a)
        data_predic

   
