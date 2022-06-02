import re
import numpy as np
from filterpy.kalman import KalmanFilter
from datetime import date
import os
import sys
class logisticRegression:
    def __init__(self,learningRate=0.001,n_iters=1000):
        self.learningRate = learningRate
        self.n_iters = n_iters
        self.weights = None
        self.bias    = None
    def fit(self, X, y,X_Validation,y_Validation) :
        # Train
        n_samples,n_features = X.shape
        # Test
        n_samplesValidation,n_featuresValidation = X_Validation.shape
        self.weights         = np.zeros(n_features)
        self.bias = 0
        lossTrain = [[],[]]
        lossValidation = [[],[]]
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
            #Get loss
                # Train
            linear_modelTrain    =   np.dot(X, self.weights) + self.bias
            y_predictedTrain     =   self.__sigmoid(linear_modelTrain)
            loss = (1/n_samples) * np.sum(np.square(y_predictedTrain-y))
            lossTrain.append(loss)
            lossTrain[0].append(loss)
            lossTrain[1].append(i)
                # Validation
            linear_modelValidation    =   np.dot(X_Validation, self.weights) + self.bias
            y_predictedValidation     =   self.__sigmoid(linear_modelValidation)           
            loss = (1/n_samplesValidation) *np.sum(np.square(y_predictedValidation-y_Validation))
            lossValidation.append(loss)
            lossValidation[0].append(loss)
            lossValidation[1].append(i)

        return lossTrain,lossValidation
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
    def saveWeights(self,path):
        try:
            os.makedirs(path)
        except:
            pass
        file = path + '/saveData.txt'
        f = open(file, "w")
        text = "Date :"+"\n"
        text += str(date.today())+"\n"
        text += "Weights :"+"\n"
        for i in (self.weights):
            text +=str(i)+"\n"
        text += "Bias :"+"\n"
        text += str(self.bias)+"\n"
        text += "End."+"\n"
        f.write(text)
        f.close()
        
    def loadWeights(self,path):
        file = path + '/saveData.txt'
        f = open(file, "r")
        text = f.read()
        weightFlag = False
        weight = []
        bias   = 0.0
        biasFlag = False
        endFlag  = False
        for line in text.split("\n"):
            if line == "Weights :":
                weightFlag = True
            elif line == "Bias :":
                biasFlag = True
            elif line == "End.":
                endFlag = True
            elif (weightFlag==True) and (biasFlag==False):
                weight.append(float(line))
            elif(biasFlag==True )   and (endFlag==False):
                bias=float(line)
        self.weights = np.array(weight)
        self.bias = bias
