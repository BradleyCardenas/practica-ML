import pandas as pd
import streamlit as st

#Ocultar el botón de más opciones

df = pd.read_csv('Iris.csv')
st.header('Dataframe original')
st.dataframe(df)

st.header('Estadisticas')
st.write('Filas, columnas:')
st.write(df.shape)

st.write('Describe')
st.dataframe(df.describe())