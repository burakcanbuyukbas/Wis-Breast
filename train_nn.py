import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt
import numpy as np
from sklearn import preprocessing
from neural_network import Layer, ActivationLayer, FullyConnectedLayer, NeuralNetwork
from sklearn.metrics import accuracy_score
from utils import tanh, tanh_prime, mse, mse_prime, dReLU, ReLU, Sigmoid, dSigmoid, plotConfusionMatrix
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential

data = pd.read_csv("data.csv", index_col=32)
#print(data.head())

data = data.drop(['id'], axis=1)
print(data.isnull().values.any())
label_encoder = preprocessing.LabelEncoder()

Y = label_encoder.fit_transform(data['diagnosis'])
X = data.drop('diagnosis', axis=1)
#print(X.columns)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=11)





nn = NeuralNetwork(loss=mse, loss_prime=mse_prime)
#nn.add_layer(FullyConnectedLayer(30, 100))
#nn.add_layer(ActivationLayer(tanh, tanh_prime))
#nn.add_layer(FullyConnectedLayer(100, 50))
#nn.add_layer(ActivationLayer(tanh, tanh_prime))
#nn.add_layer(FullyConnectedLayer(50, 1))
#nn.add_layer(ActivationLayer(tanh, tanh_prime))

nn.add_layer(FullyConnectedLayer(30, 100))
nn.add_layer(ActivationLayer(ReLU, dReLU))
nn.add_layer(FullyConnectedLayer(100, 50))
nn.add_layer(ActivationLayer(ReLU, dReLU))
nn.add_layer(FullyConnectedLayer(50, 1))
nn.add_layer(ActivationLayer(Sigmoid, dSigmoid))

nn.fit(X_train, Y_train, iter=1000, learning_rate=0.003)
Y_pred = nn.predict(X_test)
print(accuracy_score(Y_pred.round(), Y_test))
plotConfusionMatrix(Y_test, Y_pred.round())