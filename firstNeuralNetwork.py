# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:07:32 2019

@author: Lucas
"""
# Importing libraries
import numpy as np

 # Importing dataset and setting X and y variables
X_original = np.array([[3.,4.],[2.,7.],[8.,4.], [8.,8.]]) # first column shows study hours, second shows sleep hours
y_original = np.array([[5.5],[6.1],[6.4],[6.9]]) # shows test score on a scale from 0.0-7.0

# Normalizing data
X = X_original/np.amax(X_original, axis=0)
y = y_original/7.0

# Defining the architechture for the simple neural network
inputLayerSize = 2
hiddenLayerSize = 3
outputLayerSize = 1 

# Defining the activation function used in each layer (sigmoid)
def sigmoid(z, deriv=False):
    if deriv:
      return np.exp(-z)/(1+(np.exp(-z)))**2
    else:
      return 1/(1+np.exp(-z))
    
# Initializing weight matrices with normally distributed random numbers
w1 = np.random.randn(inputLayerSize, hiddenLayerSize) - 1
w2 = np.random.randn(hiddenLayerSize, outputLayerSize) - 1

# Defining a for loop for each epoch that consists in forward propagation and backpropagation
costVector = [] # an array that holds the cost values for each iteration
costNumber = 0
for i in range(50000):
  # Calculating values for each layer
  z2 = X.dot(w1)
  a2 = sigmoid(z2)  
  
  z3 = a2.dot(w2)
  y_hat = sigmoid(z3)
  
  # Calculating the cost function (C)
  error = y - y_hat
  C = sum((error)**2) / 2
  costVector.append(C)
  if (i % 10000 == 0):
    print('Cost',costNumber,':', C)
    costNumber += 1
  # Calculating the partial derivatives of the cost with respect to each of the weights (Gradient Descent)
  delta3 = (-error)* sigmoid(z3, deriv=True)
  dCdW2 = (a2.T).dot(delta3)
  
  delta2 = delta3.dot(w2.T) * sigmoid(z2, deriv=True)
  dCdW1 = (X.T).dot(delta2)
  
  # Updating the weight matrices
  w1 = w1 - (dCdW1) * 0.01
  w2 = w2 - (dCdW2) * 0.01

# Visualizing the cost of each epoch
import matplotlib.pyplot as plt
plt.plot(costVector)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost reduction')
plt.show()
