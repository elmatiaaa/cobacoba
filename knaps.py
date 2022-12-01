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
# Project Data Mining
""")

st.write("=========================================================================")

tab1, tab2, tab3, tab4 = st.tabs(["Import Data", "Preprocessing", "Modelling", "Implementasi"])
with tab1:
    st.write("Import Data")
    data = pd.read_csv("https://raw.githubusercontent.com/elmatiaaa/Machine-Learning/main/winequality-red.csv")
    st.dataframe(data)

with tab2:
    quality_series = data.loc[:, "quality"]
    quality_categorical_series = pd.cut(quality_series, [0, 5, 10], labels=["bad", "good"])
    data["quality"] = quality_categorical_series
    data
    
    x=data[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
    y=data["quality"].values
    
    scaler = MinMaxScaler()
    scaler.fit(x)
    x=scaler.transform(x)
    x
    
with tab3:
    x_train, x_test,y_train,y_test= train_test_split(x,y,random_state=0)    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    # from sklearn.feature_extraction.text import CountVectorizer
    # cv = CountVectorizer()
    # x_train = cv.fit_transform(x_train)
    # x_test = cv.fit_transform(x_test)
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")
    
    # NB
    GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(x_test)
    
    y_compare = np.vstack((y_test,y_pred)).T
    nvklasifikasi.predict_proba(x_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    # akurasi = 10

    # KNN 
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # DT

    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    # prediction
    dt.score(x_test, y_test)
    y_pred = dt.predict(x_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    if des :
        if mod :
            st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasiii))
   
with tab4:
    st.write("# Implementation")
    fixed_acidity = st.text_input('fixed acidity :')

    # GENDER
    volatile_acidity = st.text_input('volatile_acidity :')
    citric_acid = st.text_input('citric_acid :')
    residual_sugar = st.text_input('residual_sugar :')
    chlorides = st.text_input('chlorides :')
    free_sulfur_dioxide = st.text_input('free sulfur dioxide :')
    total_sulfur_dioxide = st.text_input('total sulfur dioxide :')
    density = st.text_input('density :')
    pH = st.text_input('pH :')
    sulphates = st.text_input('sulphates :')
    alcohol = st.text_input('alcohol :')
    quality = st.text_input('quality :')


    
    # Sex = st.radio(
    # "Masukkan Jenis Kelamin Anda",
    # ('Laki-laki','Perempuan'))
    # if Sex == "Laki-laki":
    #     Sex_Female = 0
    #     Sex_Male = 1
    # elif Sex == "Perempuan" :
    #     Sex_Female = 1
    #     Sex_Male = 0

    # BP = st.radio(
    # "Masukkan Tekanan Darah Anda",
    # ('Tinggi','Normal','Rendah'))
    # if BP == "Tinggi":
    #     BP_High = 1
    #     BP_LOW = 0
    #     BP_NORMAL = 0
    # elif BP == "Normal" :
    #     BP_High = 0
    #     BP_LOW = 0
    #     BP_NORMAL = 1
    # elif BP == "Rendah" :
    #     BP_High = 0
    #     BP_LOW = 1
    #     BP_NORMAL = 0

    # Cholesterol = st.radio(
    # "Masukkan Kadar Kolestrol Anda",
    # ('Tinggi','Normal'))
    # if Cholesterol == "Tinggi" :
    #     Cholestrol_High = 1
    #     Cholestrol_Normal = 0 
    # elif Cholesterol == "Normal":
    #     Cholestrol_High = 0
    #     Cholestrol_Normal = 1
        
    # Na_to_K = st.number_input('Masukkan Rasio Natrium Ke Kalium dalam Darah')



    def submit():
        # input
        inputs = np.array([[
            fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,ree_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,quality
            ]])
        # st.write(inputs)
        # baru = pd.DataFrame(inputs)
        # input = pd.get_dummies(baru)
        # st.write(input)
        # inputan = np.array(input)
        # import label encoder
        le = joblib.load("le.save")
        model1 = joblib.load("knn.joblib")
        y_pred3 = model1.predict(inputs)
        st.write(f"Berdasarkan data yang Anda masukkan, maka anda dinyatakan : {le.inverse_transform(y_pred3)[0]}")

    all = st.button("Submit")
    if all :
        st.balloons()  
        submit()

