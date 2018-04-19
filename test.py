#!/usr/bin/python

# Author: Arun George
# Class: UConn, CSE 4502, Spring 2018
# Instructor: Sanguthevar Rajasekaran
# Description: Final Project

import urllib
import pickle
import gzip
import numpy as np
from neuralnet import Network


# testfile = urllib.URLopener()
# testfile.retrieve("http://randomsite.com/file.gz", "mnist.pkl.gz")

def getDatasets():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training, validation, testing = pickle.load(f, encoding="latin1")
    f.close()

    training_inputs 	= [np.reshape(x, (784, 1)) for x in training[0]]
    # training_results 	= [vectorized_result(y) for y in training[1]]
    # training_data 		= zip(training_inputs, training_results)
    training_data 		= zip(training_inputs, training[1])

    validation_inputs 	= [np.reshape(x, (784, 1)) for x in validation[0]]
    # validation_results 	= [vectorized_result(y) for y in validation[1]]
    # validation_data 	= zip(validation_inputs, validation_results)
    validation_data 	= zip(validation_inputs, validation[1])

    testing_inputs 		= [np.reshape(x, (784, 1)) for x in testing[0]]
    # testing_results 	= [vectorized_result(y) for y in testing[1]]
    # testing_data 		= zip(testing_inputs, testing_results)
    testing_data 		= zip(testing_inputs, testing[1])

    return (training_data, validation_data, testing_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


if __name__ == '__main__':
	training, validation, testing = getDatasets()
	training = list(training)
	# Network parameters:
	#     2nd param is epochs count
	#     3rd param is batch size
	#     4th param is learning rate (eta)
	net = Network([784, 30, 10])
	net.gradientdescent(training, 30, 10, 3.0, test_data=testing)
	# net.gradientdescent(training, 30, 10, 3.0)