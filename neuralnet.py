#!/usr/bin/python

# Author: Arun George
# Class: UConn, CSE 4502, Spring 2018
# Instructor: Sanguthevar Rajasekaran
# Description: Final Project

import random
import numpy as np
import activations as actvtn

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes      = sizes
        self.biases     = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights    = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedForward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = actvtn.sigmoid(np.dot(w, a)+b)
        return a

    def gradientDescent(self, training_data, epochs, mini_batch_size, eta, test_data):
        training_data = list(training_data)
        n = len(training_data)
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.updateMiniBatch(mini_batch, eta)
            correct_classification = self.evaluate(test_data)
            print("Epoch {} : {} / {}".format(j, correct_classification, n_test));
            accuracy = correct_classification / n_test * 100
        return accuracy

    def updateMiniBatch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backPropagation(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backPropagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = actvtn.sigmoid(z)
            activations.append(activation)
        delta = self.costDerivative(activations[-1], y) * actvtn.sigmoidPrime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = actvtn.sigmoidPrime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedForward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def costDerivative(self, output_activations, y):
        return (output_activations - y)