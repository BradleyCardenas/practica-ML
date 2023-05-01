import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

df = pd.read_csv('./Iris.csv')

st.title('Gráficas')
for i in range(len(df.columns)):
    st.subheader(df.columns[i])
    fig = px.box(df, y = df.columns[i])
    st.plotly_chart(fig,use_container_width=True)


st.title('Histogramas')
for i in range(len(df.columns)):
    fig = px.histogram(df, x=df.columns[i])
    st.plotly_chart(fig, use_container_width=True)


st.title('Gráfica de correlación')
fig = px.scatter_matrix(
    df,
    dimensions = df.columns[1:4],
    color = 'Species'
)
st.plotly_chart(fig, use_container_width = True)


st.title('Gráfica de correlación - Mapa de calor')
df_numeric = df.select_dtypes(include=['float64', 'int64'])
df_corr = df_numeric.corr()

#df_corr = df.corr()
fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x = df_corr.columns,
        y = df_corr.index,
        z = np.array(df_corr)
    )
)
st.plotly_chart(fig, use_container_width = True)
