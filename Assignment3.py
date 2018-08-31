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
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'softmax', input_dim = 13))

# Adding the second hidden layer - apply grid search to find the optimal number of layers 
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'sigmoid')) 

# Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 20, nb_epoch = 100) 

# Part 3 - Making the predictions
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred = y_pred.astype(int)


# Part 4 - Evaluating, Improving and Tuning the ANN
#Evaluating the model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'softmax', input_dim = 13))
    classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'sigmoid')) 
    classifier.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score (estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1 )
mean = accuracies.mean()
variance = accuracies.std()  #lower the variance, lower the bias (overfitting)
print ('Mean: ',mean )
print ('Variance: ',variance)
 

# Tuning the ANN with grid search
from sklearn.model_selection import GridSearchCV
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size' : [10, 20, 30],
              'epochs' : [50, 100, 200],
              'optimizer' : ['adam', 'rmsprop']}
grid_search = GridSearchCV (estimator = classifier,
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print ('Best accuracy: ',best_accuracy )
print ('Best parameters: ',best_parameters)
