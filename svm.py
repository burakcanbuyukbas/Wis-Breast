import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, precision_score

class svm:
    def __init__(self, X, Y, learning_rate=0.00001, epochs=1000):
        self.X = X
        self.Y = Y
        self.epochs = epochs
        self.weights = np.zeros(X.shape[1])
        self.regularization_strength = 1000
        self.lr = learning_rate

    def compute_cost(self, W, X, Y):
        # calculate hinge loss
        N = X.shape[0]
        distances = 1 - Y * (np.dot(X, W))
        distances[distances < 0] = 0  # equivalent to max(0, distance)
        hinge_loss = self.regularization_strength * (np.sum(distances) / N)

        # calculate cost
        cost = 1 / 2 * np.dot(W, W) + hinge_loss
        return cost

    def calculate_cost_gradient(self, W, X_batch, Y_batch):
        if type(Y_batch) == np.float64:
            Y_batch = np.array([Y_batch])
            X_batch = np.array([X_batch])

        distance = 1 - (Y_batch * np.dot(X_batch, W))
        dw = np.zeros(len(W))

        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = W
            else:
                di = W - (self.regularization_strength * Y_batch[ind] * X_batch[ind])
            dw += di

        dw = dw / len(Y_batch)
        return dw

    def train(self):
        weights = np.zeros(self.X.shape[1])
        prev_cost = float("inf")
        cost_threshold = 0.0001

        for epoch in range(1, self.epochs):
            # shuffle to prevent repeating update cycles
            X, Y = shuffle(self.X, self.Y)
            for ind, x in enumerate(X):
                ascent = self.calculate_cost_gradient(weights, x, Y[ind])
                weights = weights - (self.lr * ascent)

            if epoch % 100 == 0 or epoch == self.epochs - 1:
                cost = self.compute_cost(weights, X, Y)
                print("Epoch: {} => Cost: {}".format(epoch, cost))
                # stoppage criterion
                if abs(prev_cost - cost) < cost_threshold * prev_cost:
                    return weights
                prev_cost = cost
        return weights

    def predict(self, W, X_test, Y_test):

        Y_pred = np.array([])
        for i in range(X_test.shape[0]):
            yp = np.sign(np.dot(X_test[i], W))
            Y_pred = np.append(Y_pred, yp)

        print("accuracy on test dataset: {}".format(accuracy_score(Y_test, Y_pred)))
        print("recall on test dataset: {}".format(recall_score(Y_test, Y_pred)))
        print("precision on test dataset: {}".format(recall_score(Y_test, Y_pred)))
        return Y_pred

