import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

base = pd.read_csv('/workspaces/learning-machine-learning/udemy/a_z/iris_more_classes/iris.csv')
previsores = base.iloc[:, 0:4].values
clase = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
clase = labelencoder.fit_transform(clase)
# iris setosa = 0, iris virginica = 1, iris versicolor = 2
clase_dummy = to_categorical(clase)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, clase_treinamento, clase_teste = train_test_split(previsores, clase_dummy, test_size=0.25)

entradas = 4
saidas = 3
units = (entradas+saidas) / 2

classificador = Sequential()
classificador.add(Dense(units=units, activation='relu', input_dim=entradas))
classificador.add(Dense(units=units, activation='relu'))
classificador.add(Dense(units=saidas, activation='softmax'))

classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

classificador.fit(previsores_treinamento, clase_treinamento, batch_size=10, epochs=1000)

resultado = classificador.evaluate(previsores_teste, clase_teste)
print(resultado)
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
clase_teste2 = [np.argmax(t) for t in clase_teste]
previsoes2 = [np.argmax(t) for t in previsoes]
precisao = accuracy_score(clase_teste2, previsoes2)
matriz = confusion_matrix(clase_teste2, previsoes2)
print("Precisão: ", matriz)
