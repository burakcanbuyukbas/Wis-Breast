from keras import Sequential
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt
import numpy as np
from sklearn import preprocessing
from neural_network import Layer, ActivationLayer, FullyConnectedLayer, NeuralNetwork
from sklearn.metrics import accuracy_score
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten
from sklearn.preprocessing import MinMaxScaler
from utils import plotConfusionMatrix

data = pd.read_csv("data.csv", index_col=32)
#print(data.head())

data = data.drop(['id'], axis=1)
label_encoder = preprocessing.LabelEncoder()

Y = label_encoder.fit_transform(data['diagnosis'])
X = data.drop('diagnosis', axis=1)
#print(X.columns)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)





model = Sequential()
model.add(Dense(20, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, Y_train, epochs=200)
_, accuracy = model.evaluate(X_test, Y_test, verbose=0)

Y_pred = model.predict_classes(X_test)
print(accuracy_score(Y_pred.round(), Y_test))
plotConfusionMatrix(Y_test, Y_pred.round())