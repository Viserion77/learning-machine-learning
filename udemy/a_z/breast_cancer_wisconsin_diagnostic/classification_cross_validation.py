import pandas as pd
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('/workspaces/learning-machine-learning/udemy/a_z/breast_cancer_wisconsin_diagnostic/entradas_breast.csv')
clase = pd.read_csv('/workspaces/learning-machine-learning/udemy/a_z/breast_cancer_wisconsin_diagnostic/saidas_breast.csv')

def criarRede():
    # escolher quantidade de neuronios
    entradas = 30
    saidas = 1
    units = (entradas+saidas) / 2

    classificador = Sequential()
    classificador.add(Dense(units=units, activation='relu', kernel_initializer='random_uniform', input_dim=entradas))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=units, activation='relu', kernel_initializer='random_uniform'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=units/2, activation='relu', kernel_initializer='random_uniform'))
    classificador.add(Dense(units=saidas, activation='sigmoid'))

    # classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    otimization = keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0001, clipvalue=0.5)
    classificador.compile(optimizer=otimization, loss='binary_crossentropy', metrics=['binary_accuracy'])
      
    return classificador

classificador = KerasClassifier(build_fn=criarRede, epochs=100, batch_size=10)

resultados = cross_val_score(estimator=classificador, X=previsores, y=clase, cv=10, scoring='accuracy')

media = resultados.mean()
desvio = resultados.std()