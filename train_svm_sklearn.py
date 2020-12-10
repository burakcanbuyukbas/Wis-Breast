from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
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


svc = SVC(kernel='linear')
svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

print(accuracy_score(Y_pred.round(), Y_test))
plotConfusionMatrix(Y_test, Y_pred.round())