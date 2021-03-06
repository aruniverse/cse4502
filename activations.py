#!/usr/bin/python

# Author: Arun George, Vedant Patel, and Vishal Cherian
# Class: UConn, CSE 4502, Spring 2018
# Instructor: Sanguthevar Rajasekaran
# Description: Final Project

import numpy as np

# Activation functions that were used to test the accuracy through Epoch 
def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

def sigmoidPrime(z):
	return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
	return np.exp(z) / np.sum(np.exp(z))

def softmaxPrime(z):
	return softmax(z) * (1 - softmax(z))

def tanh(z):
	return np.tanh(z)

def tanhPrime(z):
	return 1 - tanh(z) * tanh(z)
