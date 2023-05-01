import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('./../Iris.csv')

#Dividir los datos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(df[df.columns[1:5]], df[df.columns[-1]], test_size=0.2)


# Creando un array de modelos para evaluar cuál funciona mejor

## Modelo LogisticRegression
## -------------------------
modelo = LogisticRegression(random_state = 0).fit(x_train, y_train)

print(modelo.score(x_test,y_test))
print(modelo.predict(x_test))

## Cambiando el dataset de entrenamiento y pruebas para mejorar el desempeño del modelo (Crossvalidation)
kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)
resultados = cross_val_score(modelo, x_train, y_train, cv = kfold, scoring = 'accuracy')
#print(resultados)


## Modelo KNeighborsClassifier
modeloK = KNeighborsClassifier(n_neighbors = 4).fit(x_train, y_train)

print(modeloK.score(x_test, y_test))
print(modeloK.predict(x_test))