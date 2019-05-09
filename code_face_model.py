import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import os
from keras_tqdm import TQDMCallback
import tensorflow
os.environ['KERAS_BACKEND']='theano'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DNN_Model:
    def __init__(self, inputDim, hidenDim, outputDim, lr=0.02, loadModel = False, modeFile = "model0416.h5"):
        if loadModel:
            self.model = load_model(modeFile)
        else:
            self.model = Sequential()
            self.model.add(Dense(hidenDim, input_dim=inputDim, activation='sigmoid'))
            self.model.add(Dense(hidenDim*2, input_dim=inputDim, activation='sigmoid'))
            self.model.add(Dense(hidenDim, activation='sigmoid'))
            self.model.add(Dense(outputDim, activation='softmax'))
            self.model.summary()
            sgd = keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
            self.model.compile(loss='mse', optimizer=sgd, metrics=["accuracy"])

    def train(self, x, target, batch_size=500, epochs=10000, verbose=0):
        self.model.fit(x, target, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[TQDMCallback()]) # ,callbacks=[TQDMCallback()]

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def saveModel(self, fileName):
        self.model.save(fileName)

    def loadModel(self, fileName):
        self.model = load_model(fileName)



from sklearn.svm import SVC


class SVM_Model:
    def __init__(self):
        self.clf = SVC(gamma='scale', decision_function_shape='ovr')

    def train(self, x, y):
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)

    def evaluate(self, y1, y2):
        print(y1)
        print(y2)
        count = np.sum(y1 == y2)
        print("测试个数:", len(y2))
        print("正确个数:", count)
        print("正确率:", count/len(y2))
        # return count


