import numpy as np
from utils import ReLU, dReLU, Sigmoid, dSigmoid
import pickle

class NeuralNetwork:

    def __init__(self, X, Y):

        #X: Holds the input layer, the data given to the network.
        self.X = X

        #Y: Holds the desired output
        self.Y = Y

        #Yh: Holds the output that our network produces.
        self.Yh = np.zeros((1, self.Y.shape[1]))

        #dims: number of neurons in each layer
        self.dims = [60, 120, 1]

        #param: dictionary that holds W and b parameters of each of the layers of the network
        self.param = {}

        #cache: dictionary that holds some intermediate calculations needed during the backward pass
        self.ch = {}

        #loss: the distance between Yh and Y.
        self.loss = []

        #lr: learning rate
        self.lr = 0.0003

        #iter: iteration count
        self.iter = 10000


    def nInit(self):
        np.random.seed(1)
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0])
        self.param['b1'] = np.zeros((self.dims[1], 1))
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1])
        self.param['b2'] = np.zeros((self.dims[2], 1))
        return

    def forward(self):
        Z1 = self.param['W1'].dot(self.X) + self.param['b1']
        A1 = ReLU(Z1)
        self.ch['Z1'], self.ch['A1'] = Z1, A1

        Z2 = self.param['W2'].dot(A1) + self.param['b2']
        A2 = Sigmoid(Z2)
        self.ch['Z2'], self.ch['A2'] = Z2, A2

        self.Yh = A2
        loss = self.nloss(A2)
        return self.Yh, loss

    def nloss(self, Yh):
        loss = (1. / self.Y.shape[1]) * (-np.dot(self.Y, np.log(Yh).T) - np.dot(1 - self.Y, np.log(1 - Yh).T))
        return loss

    def backward(self):
        dLoss_Yh = - (np.divide(self.Y, self.Yh) - np.divide(1 - self.Y, 1 - self.Yh))

        dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z2'])
        dLoss_A1 = np.dot(self.param["W2"].T, dLoss_Z2)
        dLoss_W2 = 1. / self.ch['A1'].shape[1] * np.dot(dLoss_Z2, self.ch['A1'].T)
        dLoss_b2 = 1. / self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1], 1]))
#
        dLoss_Z1 = dLoss_A1 * dReLU(self.ch['Z1'])
        dLoss_A0 = np.dot(self.param["W1"].T, dLoss_Z1)
        dLoss_W1 = 1. / self.X.shape[1] * np.dot(dLoss_Z1, self.X.T)
        dLoss_b1 = 1. / self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1], 1]))

        self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
        self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2

    def predict(self, x, y):
        self.X = x
        self.Y = y
        comp = np.zeros((1, x.shape[1]))
        pred, loss = self.forward()

        for i in range(0, pred.shape[1]):
            if pred[0, i] > 0.5:
                comp[0, i] = 1
            else:
                comp[0, i] = 0

        print("Acc: " + str(np.sum((comp == y) / x.shape[1])))

        return comp

    def fit(self, X, Y, iter=3000):
        np.random.seed(1)

        self.nInit()
        self.iter = iter

        for i in range(0, iter):
            Yh, loss = self.forward()
            self.backward()
            if i % 500 == 0:
                print("Cost after iteration %i: %f" % (i, loss))
                self.loss.append(loss)

        #plt.plot(np.squeeze(self.loss))
        #plt.ylabel('Loss')
        #plt.xlabel('Iter')
        #plt.title("Lr =" + str(self.lr))
        #plt.show()

    def save_model(self, name=None):
        data = {
            'W1': (self.param['W1']),
            'b1': (self.param['b1']),
            'W2': (self.param['W2']),
            'b2': (self.param['b2'])
        }
        if name == None:
           name = ('model_lr' + str(self.lr).replace('.', ',') + '_iter' + str(self.iter)
            + '_dims' + str(self.dims[0]) + '-' + str(self.dims[1]) + '-' + str(self.dims[2]))
        path = (r'models/' + name)
        outfile = open(path, 'wb')
        pickle.dump(data, outfile)
        outfile.close()

    def load_model_and_predict(self, name, x, y):
        infile = open(name, 'rb')
        data = pickle.load(infile)
        infile.close()
        W1 = data['W1']
        b1 = data['b1']
        W2 = data['W2']
        b2 = data['b2']

        comp = np.zeros((1, x.shape[1]))
        Z1 = W1.dot(x) + b1
        A1 = ReLU(Z1)

        Z2 = W2.dot(A1) + b2
        A2 = Sigmoid(Z2)

        loss = self.nloss(A2)
        pred, loss = A2, loss
        for i in range(0, pred.shape[1]):
            if pred[0, i] > 0.5:
                comp[0, i] = 1
            else:
                comp[0, i] = 0

        print("Acc: " + str(np.sum((comp == y) / x.shape[1])))

        return comp





if False:
    #Neural_Network(X_train,X_test, Y_train, Y_test)

    print(np.transpose(X_train).shape)
    print(np.transpose(Y_train.reshape(len(Y_train), 1)).shape)

    # custom neural network
    nn = neural_network(np.transpose(X_train), np.transpose(Y_train.reshape(len(Y_train), 1)))
    nn.lr = 0.0025
    nn.dims = [30, 30, 1]

    # train
    nn.fit(np.transpose(X_train), np.transpose(Y_train.reshape(len(Y_train), 1)), iter=10000)
    Y_pred = nn.predict(np.transpose(X_test), np.transpose(Y_test.reshape(len(Y_test), 1)))
    nn.save_model()

if False:
    nn = neural_network(np.transpose(X_test), np.transpose(Y_test.reshape(len(Y_test), 1)))

    Y_pred = nn.load_model_and_predict(r'models/model_lr0,0025_iter10000_dims30-30-1', np.transpose(X_test), np.transpose(Y_test.reshape(len(Y_test), 1)))

    print(accuracy_score(np.squeeze(Y_pred), np.transpose(Y_test)))
