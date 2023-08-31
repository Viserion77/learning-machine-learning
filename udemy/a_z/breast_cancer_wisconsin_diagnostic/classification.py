import pandas as pd

previsores = pd.read_csv('/workspaces/learning-machine-learning/udemy/a_z/breast_cancer_wisconsin_diagnostic/entradas_breast.csv')
clase = pd.read_csv('/workspaces/learning-machine-learning/udemy/a_z/breast_cancer_wisconsin_diagnostic/saidas_breast.csv')

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, clase_treinamento, clase_teste = train_test_split(previsores, clase, test_size=0.25)

import keras
from keras.models import Sequential
from keras.layers import Dense

# escolher quantidade de neuronios
entradas = 30
saidas = 1
units = (entradas+saidas) / 2

classificador = Sequential()
classificador.add(Dense(units=units, activation='relu', kernel_initializer='random_uniform', input_dim=entradas))
classificador.add(Dense(units=units, activation='relu', kernel_initializer='random_uniform'))
classificador.add(Dense(units=units/2, activation='relu', kernel_initializer='random_uniform'))
classificador.add(Dense(units=saidas, activation='sigmoid'))

# classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
otimization = keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0001, clipvalue=0.5)
classificador.compile(optimizer=otimization, loss='binary_crossentropy', metrics=['binary_accuracy'])

classificador.fit(previsores_treinamento, clase_treinamento, batch_size=10, epochs=100)

pesos0 = classificador.layers[0].get_weights()
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()

print(pesos0)
print(pesos1)
print(pesos2)

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(clase_teste, previsoes)
matriz = confusion_matrix(clase_teste, previsoes)

resultado = classificador.evaluate(previsores_teste, clase_teste)

print("Precisão: ", precisao)
print("Matriz: ", matriz)
print("Resultado: ", resultado)
