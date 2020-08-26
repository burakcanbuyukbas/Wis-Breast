import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from utils import plotConfusionMatrix
from sklearn.preprocessing import MinMaxScaler
from svm import svm as SVM

data = pd.read_csv("data.csv", index_col=32)
#print(data.head())

data = data.drop(['id'], axis=1)
print(data.isnull().values.any())
label_encoder = preprocessing.LabelEncoder()

Y = label_encoder.fit_transform(data['diagnosis']).astype(np.float64)
Y = np.where(Y == 0, -1, Y)
X = data.drop('diagnosis', axis=1)
#print(X.columns)




scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=11)

svm = SVM(X_train, Y_train, 0.0000005, 100000)
W = SVM.train(svm)
Y_pred = svm.predict(W, X_test, Y_test)
plotConfusionMatrix(Y_test, Y_pred.round())

