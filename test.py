#!/usr/bin/python

# Author: Arun George, Vedant Patel, and Vishal Cherian
# Class: UConn, CSE 4502, Spring 2018
# Instructor: Sanguthevar Rajasekaran
# Description: Final Project

import os
import urllib.request
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt 
from enum import Enum
import neuralnet

class Activations(Enum):
    SIGMOID = 1
    SOFTMAX = 2
    TANH    = 3


def getDatasets():
    if not os.path.exists('./mnist.pkl.gz') :
        urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")

    f = gzip.open('mnist.pkl.gz', 'rb')
    training, validation, testing = pickle.load(f, encoding="latin1")
    f.close()

    training_inputs 	= [np.reshape(x, (784, 1)) for x in training[0]]
    training_results 	= [vectorizedResult(y) for y in training[1]]
    training_data 		= zip(training_inputs, training_results)

    validation_inputs 	= [np.reshape(x, (784, 1)) for x in validation[0]]
    validation_data 	= zip(validation_inputs, validation[1])

    testing_inputs 		= [np.reshape(x, (784, 1)) for x in testing[0]]
    testing_data 		= zip(testing_inputs, testing[1])

    return (training_data, validation_data, testing_data)

def vectorizedResult(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# Various kinds of datasets
def main():
    training, validation, testing = getDatasets()
    training   = list(training)
    validation = list(validation)
    testing    = list(testing)


    # Test vector; all the functions will go through for each of the elements inside alpharates array
    alphaRates  = [0.001, 0.01, 0.1, 1, 3, 5, 10, 25, 50] 
    for i in range(1, 4):
        name = Activations(i).name
        print(name)
        logFile = 'log_{}'.format(name)
        with open(logFile, 'w') as lf :
            pass    # flush the file everytime we run the tests
            lf.write(name)
            lf.write("\n")
        lf.close()

        accuracyRates = []
        net = neuralnet.Network([784, 30, 10], i)
        for alphaIndex, alphaRate in enumerate(alphaRates):
            print("Learning rate = {} ".format(alphaRate))
            with open(logFile, 'a+') as lf :    # append to the log file test data 10 times for each length
                lf.write("\tLearning rate = {} ".format(alphaRate))
                lf.write("\n")
            lf.close()
            accuracy = net.gradientDescent(training, 30, 10, alphaRate, validation)
            accuracyRates.append(accuracy)

        # Graph Plotting
        plt.figure()
        plt.xlabel('Learning Rate')
        plt.ylabel('Accuracy')
        plt.title('Neural Network')
        plt.plot(alphaRates, accuracyRates)
        plt.axis([0, 50, 0, 100])
        # plt.show()
        # saving picture to your current directory 
        plt.savefig('NN_{}'.format(name))

if __name__ == '__main__':
    main()
