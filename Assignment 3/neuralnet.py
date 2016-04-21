#Author - Anmol Mohanty
#Date :: 4/19

import math
import sys
import decimal
import random

import numpy as np

from scipy.io.arff import loadarff

class NN:

    def __init__(self,input_count):
        self.weights=[] #declaring list of weights
        for index in range(0,input_count) #for i=0>input_count
            self.weights.append(0.1)
        self.bias=0.1





    #sigmoid function
    def sigmoid(self,x):
	   return 1/(1+np.exp(x))

    def dot(self,x,y):
        dot_product = 0.0
        for index in range(0,len(x)):
            dot_product+=float(x[index]*y[index]) #dot product sum of float casted
        return dot_product

    def activation(self,x):
        return self.sigmoid(self.dot(x,self.weights)+self.bias) #sigma(weights*input+bias)

    def predictor(self,instance):
        return self.threshold(self.activation(instance))

    def threshold(self,x):
	   if x > 0.5
		  return 1 #class 2
	   else:
		  return 0 #class 1

    #main trainer for the perceptron
    def trainer(self,tr_set,lr,nepochs):
        for current_epoch in range(0,nepochs)

	       for instance in tr_set:
	       	#1.find the output of this training instance
    		#2.find the error derivative due to this instance
		    #3.update the weights

		      error=[] #error list
		      expected_output=instance[-1] #last element gives the class label



if pass __name__ == "__main__":

    #training file pulled from cmd
    tf = sys.argv[1]
    nfolds=sys.argv[2]
    lr=sys.argv[3]
    nepochs=sys.argv[4]


    #file I/O
    #open the file in read mode
    tfp=open(tf,"r")
    train_data, train_meta = loadarff(tfp)
    #opclass=train_data[163][-1]
    print len(train_data)
    print sigmo(train_data[1][1])
    #print opclass


