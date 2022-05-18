import re
import numpy as np
class logisticRegression:
    def __init__(self,learningRate=0.001,n_iters=1000):
        self.learningRate = learningRate
        self.n_iters = n_iters
        self.weights = None
        self.bias    = None
    def fit(self, X, y) :
        n_samples,n_features = X.shape
        self.weights         = np.zeros(n_features)
        self.bias = 0
        for i in range(self.n_iters):
            #Predic value
            linear_model    =   np.dot(X, self.weights) + self.bias
            y_predicted     =   self.__sigmoid(linear_model)
            #Update weights
            dw  = (1/n_samples)*np.dot(X.T, (y_predicted - y))
            #Update bias
            db  = (1/n_samples)*np.sum(y_predicted - y)
            self.weights    -= self.learningRate *  dw
            self.bias       -= self.learningRate *  db

    def predict(self, X,predic_class = True):
            linear_model    =  np.dot(X, self.weights) + self.bias
            y_predicted     = self.__sigmoid(linear_model)
            if (predic_class):
                return [1 if i > 0.5 else 0 for i in y_predicted]
            else:   
                return y_predicted
    def __sigmoid(self,x):
        # Activation function for logisticRegression
        return 1 / (1 + np.exp(-x))