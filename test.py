#!/usr/bin/python

# Author: Arun George
# Class: UConn, CSE 4502, Spring 2018
# Instructor: Sanguthevar Rajasekaran
# Description: Final Project

import os
import urllib.request
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt 
import neuralnet

def getDatasets():
    if not os.path.exists('./mnist.pkl.gz') :
        urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")

    f = gzip.open('mnist.pkl.gz', 'rb')
    training, validation, testing = pickle.load(f, encoding="latin1")
    f.close()

    training_inputs 	= [np.reshape(x, (784, 1)) for x in training[0]]
    training_results 	= [vectorized_result(y) for y in training[1]]
    training_data 		= zip(training_inputs, training_results)

    validation_inputs 	= [np.reshape(x, (784, 1)) for x in validation[0]]
    validation_data 	= zip(validation_inputs, validation[1])

    testing_inputs 		= [np.reshape(x, (784, 1)) for x in testing[0]]
    testing_data 		= zip(testing_inputs, testing[1])

    return (training_data, validation_data, testing_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def main():
    training, validation, testing = getDatasets()
    training   = list(training)
    validation = list(validation)
    testing    = list(testing)

    accuracyRates = []
    alphaRates  = [0.001, 0.01, 0.1, 1, 3, 5, 10, 25, 100] 
    net = neuralnet.Network([784, 30, 10])
    for alphaIndex, alphaRate in enumerate(alphaRates):
        print("Learning rate = {} ".format(alphaRate))
        accuracy = net.gradientDescent(training, 30, 10, alphaRate, validation)
        # accuracy = net.gradientDescent(training, 30, 10, 3.0, testing)
        accuracyRates.append(accuracy)

    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Neural Network')
    plt.plot(alphaRates, accuracyRates)
    plt.axis([0, 5, 0, 100])
    plt.show()

if __name__ == '__main__':
    main()
