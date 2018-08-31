# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 13:56:58 2018

@author: Mamedov
"""
#Importing the libraries
import pandas as pd
import numpy as np

#Importing/combining the datasets (US data only)
data_cleveland = pd.read_csv('processed.cleveland.data.csv', header=None, na_values=["?"])
data_va = pd.read_csv('processed.va.data.csv', header=None, na_values=["?"])
data_cleveland.columns = ['age', 'sex', 'cp','trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'the predicted attribute']
data_va.columns = data_cleveland.columns
frames = [data_cleveland, data_va]
data = pd.concat(frames)

#Imputing the missing values
data = data.fillna(data.mean())

#Converting to numpy array
X = data.drop('the predicted attribute', axis = 1).values
y = data['the predicted attribute'].values
y = y.reshape(-1,1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential #create the ANN (basic structure) - flow of the ANN
from keras.layers import Dense #build hidden layers in between

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu', input_dim = 13))

# Adding the second hidden layer - apply grid search to find the optimal number of layers 
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'sigmoid')) 

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100) 

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = np.rint(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)
