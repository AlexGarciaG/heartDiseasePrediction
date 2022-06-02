import numpy as np
from matplotlib import pyplot as plt
import pickle
import os

from sklearn.metrics import accuracy_score
class cnn:
    def __init__(self,inputLayer=None,loadModel = None):
        if loadModel is None:
            self.weights    = []
            self.bias       = []
            self.inputLayer = inputLayer
            self.layers     = -1
        else:
            file = loadModel + '/cnnModel.pkl'
            f = open(file, 'rb')
            tmp_dict = pickle.load(f)
            f.close()          
            self.__dict__.update(tmp_dict) 
    def addLayer (self,neurons):
        if len(self.weights) == 0:
            self.weights.append(np.random.rand(neurons, self.inputLayer)    - 0.5)
            self.bias.append(   np.random.rand(neurons, 1)                  - 0.5)
            self.layers+=1
        else:
            previousNeurons,previouspreviousNeurons = self.weights[len(self.weights)-1].shape
            self.weights.append(np.random.rand(neurons, previousNeurons)    - 0.5)
            self.bias.append(   np.random.rand(neurons, 1)                  - 0.5)
            self.layers+=1
        
    def __ReLU(self,Z):
        return np.maximum(Z, 0)
    def __softmax(self,Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A
    def __forward_prop(self,X):
        #First prediction
        z = [self.weights[0].dot(X)+self.bias[0]]
        a = [self.__ReLU(z[0])]
        for i in range(1,self.layers):
            #Linear regresion of the neuron
            z.append(self.weights[i].dot(a[i-1])+self.bias[i])
            #Output of the neuron
            a.append(self.__ReLU(z[i]))
        #Output layer
            #Linear regresion of the neuron
        z .append(self.weights[self.layers].dot(a[self.layers-1])+self.bias[self.layers])
            #Output of the neuron
        a .append(self.__softmax(z[self.layers]))      

        return z,a

    def __ReLU_deriv(self,Z):
        return Z > 0

    def __one_hot(self,Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def __backward_prop(self,z,a, X, Y):
        one_hot_Y = self.__one_hot(Y)
        
        #last layers
        dZ = [a[self.layers] - one_hot_Y]
        dW = [1 / self.m * dZ[0].dot(a[self.layers-1].T)]
        db = [1 / self.m * np.sum(dZ[0])]
        #Midle layers
        for i in reversed(range(1,self.layers)):
            dZ.insert(0,self.weights[i+1].T.dot(dZ[0]) * self.__ReLU_deriv(z[i]))
            dW.insert(0, 1 / self.m * dZ[0].dot(a[i-1].T))
            db.insert(0, 1 / self.m * np.sum(dZ[0]))
        #First layer
        dZ.insert(0,self.weights[1].T.dot(dZ[0]) * self.__ReLU_deriv(z[0]))
        dW.insert(0, 1 / self.m * dZ[0].dot(X.T))
        db.insert(0, 1 / self.m * np.sum(dZ[0]))

        return dW,db
    def __update_params(self, dW, db, alpha):
        for i in range(self.layers):
            self.weights[i] = self.weights[i]   - alpha * dW[i]
            self.bias[i]    = self.bias[i]      - alpha * db[i]      
    def __get_prediction(self,A2):
        return np.argmax(A2, 0)

    def get_accuracy(self,predictions, Y):
        return np.sum(predictions == Y) / Y.size
  
    def fit(self,X, Y, alpha, iterations,printEpoch=False, validationSplit = 0):
        inputSize,dataSize =  X.shape
        self.m = dataSize


        X_training    = X[:,:int(dataSize*(1.0-validationSplit))] #[1, 2, 3, 4, 5, 6, 7, 8]
        X_validation  = X[:,-int(dataSize*validationSplit):] #[10]
        Y_training    = Y[:int(dataSize*(1.0-validationSplit))] #[1, 2, 3, 4, 5, 6, 7, 8]
        Y_validation  = Y[-int(dataSize*validationSplit):] #[10]
              
        accuracy_score_Train = []
        accuracy_score_Test  = []
        inputSize,dataSize =  X_training.shape
        self.m = dataSize

        for i in range(iterations):
            z, a = self.__forward_prop(X_training)
            dW, db = self.__backward_prop(z, a, X_training, Y_training)
            self.__update_params(dW,db, alpha)
            #Accuracy
                # Train
            predictions = self.__get_prediction(a[self.layers])
            accuracy_score_Train.append(self.get_accuracy(predictions, Y_training))
                #Validation
            predictions=self.make_prediction(X_validation)
            accuracy_score_Test.append(self.get_accuracy(predictions, Y_validation))
            if (i % 10 == 0) and (printEpoch):
                print("Iteration: ", i)
                print(accuracy_score_Train[i])


        print("Final accuracy: ")
        print(accuracy_score_Train[i])
        return {'train_accuracy':accuracy_score_Train,'test_accuracy':accuracy_score_Test}

    def make_prediction(self,X):
        z,a = self.__forward_prop(X)
        predictions = self.__get_prediction(a[self.layers])
        return predictions

    def test_prediction(self,index,X,Y):
        prediction = self.make_prediction(X[:, index, None])
        label = Y[index]
        print("Prediction: ", prediction)
        print("Label: ", label)
    def saveModel(self,path):
        try:
            os.makedirs(path)
        except:
            pass

        file = path + '/cnnModel.pkl'
        with open(file, 'wb') as outp:
            pickle.dump(self.__dict__, outp, pickle.HIGHEST_PROTOCOL)


