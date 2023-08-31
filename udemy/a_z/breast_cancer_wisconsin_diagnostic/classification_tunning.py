import pandas as pd
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('/workspaces/learning-machine-learning/udemy/a_z/breast_cancer_wisconsin_diagnostic/entradas_breast.csv')
clase = pd.read_csv('/workspaces/learning-machine-learning/udemy/a_z/breast_cancer_wisconsin_diagnostic/saidas_breast.csv')

def criarRede(otimization, loss, kernel_initializer, activation, neurons):

    classificador = Sequential()
    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=30))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    classificador.add(Dense(units=1, activation='sigmoid'))

    classificador.compile(optimizer=otimization, loss=loss, metrics=['binary_accuracy'])
      
    return classificador

classificador = KerasClassifier(build_fn=criarRede)
parametros = {'batch_size': [10, 30],
                'epochs': [50, 100],
                'otimization': ['adam', 'sgd'],
                'loss': ['binary_crossentropy', 'hinge'],
                'kernel_initializer': ['random_uniform', 'normal'],
                'activation': ['relu', 'tanh'],
                'neurons': [16, 8]}

grid_search = GridSearchCV(estimator=classificador, param_grid=parametros, scoring='accuracy', cv=5)
grid_search = grid_search.fit(previsores, clase)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_