from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

masque = 6*[False]
masque[5] = True
ohe = OneHotEncoder(categorical_features  = masque, sparse=False )
le = preprocessing.LabelEncoder()

class CustomizableNN:

    def __init__(self,input_size,learning_rate=0.5,Hidden_Layers=None):
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.model = Sequential()
        self.model.add(Dense(15, input_dim=self.input_size, activation='relu'))
        # Hidden layer with 24 nodes
        if (Hidden_Layers == None):
            self.model.add(Dense(30, activation='relu'))
            self.model.add(Dense(30, activation='relu'))

        # Output Layer with # of actions: 2 nodes (left, right)
        self.model.add(Dense(2, activation='linear'))

        # Create the model based on the information above
        self.model.compile(loss='hinge',
                           optimizer=Adam(lr=self.learning_rate), metrics=["accuracy"])


    def construct(self,Hidden_Layers=None):
        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(4) and Hidden Layer with 24 nodes
        self.model.add(Dense(15, input_dim=self.input_size, activation='relu'))
        # Hidden layer with 24 nodes
        if(Hidden_Layers == None):
            self.model.add(Dense(30, activation='relu'))
            self.model.add(Dense(30, activation='relu'))

        # Output Layer with # of actions: 2 nodes (left, right)
        self.model.add(Dense(2, activation='linear'))

        # Create the model based on the information above
        self.model.compile(loss='hinge',
                      optimizer=Adam(lr=self.learning_rate),metrics=["accuracy"])


    def fit(self,X,Y):
        self.model.fit(X,Y)

    def predict(self,x):
        return self.model.predict(x)
