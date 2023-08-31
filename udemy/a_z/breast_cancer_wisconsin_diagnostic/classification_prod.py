import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense

previsores = pd.read_csv('/workspaces/learning-machine-learning/udemy/a_z/breast_cancer_wisconsin_diagnostic/entradas_breast.csv')
clase = pd.read_csv('/workspaces/learning-machine-learning/udemy/a_z/breast_cancer_wisconsin_diagnostic/saidas_breast.csv')

# escolher quantidade de neuronios
entradas = 30
saidas = 1
units = (entradas+saidas) / 4

classificador = Sequential()
classificador.add(Dense(units=units, activation='relu', kernel_initializer='normal', input_dim=entradas))
classificador.add(Dense(units=units, activation='relu', kernel_initializer='normal'))
classificador.add(Dense(units=units, activation='relu', kernel_initializer='normal'))
classificador.add(Dense(units=saidas, activation='sigmoid'))

# classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
otimization = keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0001, clipvalue=0.5)
classificador.compile(optimizer=otimization, loss='binary_crossentropy', metrics=['binary_accuracy'])

classificador.fit(previsores, clase, batch_size=10, epochs=100)

novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363]])
previsao = classificador.predict(novo)

classificador_json = classificador.to_json()
with open('/workspaces/learning-machine-learning/udemy/a_z/breast_cancer_wisconsin_diagnostic/classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('/workspaces/learning-machine-learning/udemy/a_z/breast_cancer_wisconsin_diagnostic/classificador_breast.h5')
print('Salvo modelo em disco')

from keras.models import model_from_json

arquivo = open('/workspaces/learning-machine-learning/udemy/a_z/breast_cancer_wisconsin_diagnostic/classificador_breast.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('/workspaces/learning-machine-learning/udemy/a_z/breast_cancer_wisconsin_diagnostic/classificador_breast.h5')
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

novo2 = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363]])
previsao2 = classificador.predict(novo2)
previsao2 = (previsao2 > 0.5)

print(previsao2)
