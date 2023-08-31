import pandas as pd

base = pd.read_csv('/workspaces/learning-machine-learning/udemy/a_z/autos_regrecao/autos.csv', encoding = 'ISO-8859-1')
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)

print(base['name'].value_counts())
base = base.drop('name', axis = 1)
print(base['seller'].value_counts())
base = base.drop('seller', axis = 1)
print(base['offerType'].value_counts())
base = base.drop('offerType', axis = 1)

i1 = base.loc[base.price <= 10]
base = base[base.price > 10]
i2 = base.loc[base.price > 350000]
base = base.loc[base.price < 350000]

base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() # limousine
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() # manuell
base.loc[pd.isnull(base['model'])]
base['model'].value_counts() # golf
base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() # benzin
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() # nein

valores = {'vehicleType': 'limousine', 'gearbox': 'manuell', 'model': 'golf', 'fuelType': 'benzin', 'notRepairedDamage': 'nein'}
base = base.fillna(value = valores)

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:, 10])

onehotencoder = ColumnTransformer(transformers = [('OneHot', OneHotEncoder(), [0, 1, 3, 5, 8, 9, 10])], remainder = 'passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

entrada = 316
saida = 1
units = (entrada + saida) / 2

from keras.models import Sequential
from keras.layers import Dense

regressor = Sequential()
regressor.add(Dense(units = units, activation = 'relu', input_dim = entrada))
regressor.add(Dense(units = units, activation = 'relu'))
regressor.add(Dense(units = saida, activation = 'linear'))
regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['mean_absolute_error'])

regressor.fit(previsores, preco_real, batch_size = 300, epochs = 100)

previsoes = regressor.predict(previsores)
preco_real.mean()
previsoes.mean()

