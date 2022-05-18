import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from logisticRegression import logisticRegression
import matplotlib.pyplot as plt

bc = datasets.load_breast_cancer()
x,y = bc.data, bc.target
print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1234)
def accuracy (y_true,y_pred):
    accuracy = np.sum(y_true==y_pred)/len(y_true)
    return accuracy
regressor = logisticRegression(learningRate=0.0001,n_iters=2)
regressor.fit(x_train,y_train)
predictions=regressor.predict(x_test)
print("LR classification accuracy:",accuracy(y_test,predictions))