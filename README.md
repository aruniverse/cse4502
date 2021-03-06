# CSE 4502 Project

## Overview
* Install all the requirement packages from requirements.txt
* Program takes a random sample out of 10000 examples from the MNIST dataset and feeds it through a feed-forward neural-network (FNN). For each epoch, of which there are 30, a training set is fed into the FNN that allows it to learn, and then a validation set is fed in which the newly-learned FNN will guess on. The accuracy is printed at the end of each epoch in terms of the correct examples out of total examples in the validation set.
* For each set of 30 epochs, we vary both activation function (Sigmoid, RELU, and Softmax) and the learning rates used during the gradient descent stage. The results are then plotted to show the accuracy of each configuration, along with comparing the rates at which the FNN learns with each of the configurations.

## How to Run
* Run the following command:
 ```
 python test.py
 ```
## Requirements
* Python3 and beyond
* Refer to __requirements.txt__ file 
 ```
 pip install -r requirements.txt
 ```

## References 
* It is largely based on "Neural Networks and Deep Learning" book from Michael Nielsen
