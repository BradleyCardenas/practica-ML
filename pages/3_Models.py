import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

df = pd.read_csv('./Iris.csv')

#Dividir los datos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(df[df.columns[1:5]], df[df.columns[-1]], test_size=0.2)

#Creando el header de la página en Streamlit
st.header('Modelos de ML')

## Modelo LogisticRegression
st.subheader('LogisticRegression')
modelo = LogisticRegression(random_state = 0).fit(x_train, y_train)

st.write('Score: ', modelo.score(x_test,y_test))
lr_col1, lr_col2 = st.columns([3, 1])
lr_col1.write('Predicción: ')
lr_col1.write(modelo.predict(x_test))
lr_col2.write('Resultados esperados: ')
lr_col2.write(y_test)


## Cambiando el dataset de entrenamiento y pruebas para mejorar el desempeño del modelo (Crossvalidation)
kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)
resultados = cross_val_score(modelo, x_train, y_train, cv = kfold, scoring = 'accuracy')
#print(resultados)


## Modelo KNeighborsClassifier
st.subheader('KNeighborsClassifier')
modeloK = KNeighborsClassifier(n_neighbors = 4).fit(x_train, y_train)

st.write('Score: ', modeloK.score(x_test, y_test))
kn_col1, kn_col2 = st.columns([3, 1])
kn_col1.write('Predicción: ')
kn_col1.write(modeloK.predict(x_test))
kn_col2.write('Resultados esperados: ')
kn_col2.write(y_test)
