import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

base = pd.read_csv('/workspaces/learning-machine-learning/udemy/a_z/iris_more_classes/iris.csv')
previsores = base.iloc[:, 0:4].values
clase = base.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
clase = labelencoder.fit_transform(clase)
# iris setosa = 0, iris virginica = 1, iris versicolor = 2
clase_dummy = to_categorical(clase)

def criar_rede():
    classificador = Sequential()
    classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
    classificador.add(Dense(units = 4, activation = 'relu'))
    classificador.add(Dense(units = 3, activation = 'softmax'))
    classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criar_rede, epochs = 1000, batch_size = 10)
resultados = cross_val_score(estimator = classificador, X = previsores, y = clase_dummy, cv = 10, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()
print("Media: ", media)