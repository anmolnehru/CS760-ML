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
        for index in range(0,input_count): #for i=0>input_count
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
	   if x > 0.5:
		  return 1 #class 2
	   else:
		  return 0 #class 1

    #main trainer for the perceptron
    def trainer(self,tr_set,lr,nepochs):
        for current_epoch in range(0,nepochs):
            for instance in tr_set:
                #1. find o/p of instance
                #2. find error derivative of each weight
                #3. update weights
                activation_value=self.activation(instance[:-1])
                error=[] #list for errors
                expected_output=instance[-1]
                sigmoid_value=activation_value*(1-activation_value)#figure out this
                self.bias+=lr*(expected_output-activation_value)*sigmoid_value*1.0
                for index in range(0,len(self.weights)):
                    x_i=instance[index]
                    error.append((expected_output-activation_value)*sigmoid_value*x_i)
                    self.weights[index]+=lr*error[index]

def data_partitioner(train_data,n):
    train_data=train_data[:] #figure out what's going on
    partitions=[]
    pos_instances=[]
    neg_instances=[]
    instance_fold_mapping={}

    #instance stratification
    for instance in train_data:
        if instance[-1]==1:
            pos_instances.append(instance)
        else:
            neg_instances.append(instance)



    #print "Total + count : "+str(len(pos_instances))
    #print "Total - count : "+str(len(neg_instances))


    #partititioning
    navg_pos_instances=len(pos_instances)/n
    navg_neg_instances=len(neg_instances)/n

    for partitition_count in range(0,n):
        #1.get number of + instances
        #2. remove them from the + training instances
        #3, insert in correct position
        rand_p_instances=random.sample(pos_instances, navg_pos_instances)
        for instance in rand_p_instances:
            instance_fold_mapping[str(instance)]=partitition_count+1
            pos_instances.remove(instance)
        partitions.append(rand_p_instances)


    partitition_count=0

    while len(pos_instances)!=0:
        #Keep adding to remaining partitions until you end up with empty list
        partitions[partitition_count].append(pos_instances[0])
        instance_fold_mapping[str(pos_instances[0])]=partitition_count+1
        pos_instances=pos_instances[1:]
        partitition_count=(partitition_count+1)%n


    for partitition_count in range(0,n):
        #1.get number of - instances
        #2. remove them from the - training instances
        #3, insert in correct position
        rand_n_instances=random.sample(pos_instances, navg_pos_instances)
        for instance in rand_n_instances:
            instance_fold_mapping[str(instance)]=partitition_count+1
            neg_instances.remove(instance)
            partitions.append(rand_p_instances)


    partitition_count=0

    while len(neg_instances)!=0:
        #Keep adding to remaining partitions until you end up with empty list
        partitions[partitition_count].append(neg_instances[0])
        instance_fold_mapping[str(pos_instances[0])]=partitition_count+1
        neg_instances=pos_instances[1:]
        partitition_count=(partitition_count+1)%n


    return partitions, instance_fold_mapping


def strat_cross_vald(partitions_list,l,e):

    #some definitions
    av_train_accuracy=0.0
    av_test_accuracy=0.0
    instance_output_mapping={} #dict :: key : an instance, value:[predicted o/p,expected o/p,confidence]
    n=len(partitions_list)

    for iteration_count in range(0,n):
        training_partitions=[]
        for i in range(iteration_count,iteration_count+n-1):
            if((i+1)>n):
                partitition_number=(i+1)%n
            else:
                partitition_number=(i+1)

        training_partitions.append(partitition_number-1)

        valid_partition_numbers=list(range(0,n))
        for number in training_partitions:
            if number in training_partitions:
                valid_partition_numbers.remove(number)


        test_partition_number=valid_partition_numbers[0]
        training_data=[]
        for partitition_number in training_partitions:
            for instance in partitions_list[partitition_number]:
                training_data.append(instance)


        testing_data=partitions_list[test_partition_number]
        if(n==1):
            training_data=testing_data

        neural_network=NN(len(ordered_features))
        random.shuffle(training_data)
        neural_network.trainer(training_data,l,e)

        test_accuracy=0.0

        for instance in testing_data:
            expected_output=features_values['Class'][int(instance[-1])]
            predicted_output=features_values['Class'][int(neural_network.predictor(instance[:-1]))]
            confidence_value=neural_network.activation(instance[:-1])
            instance_output_mapping[str(instance)]=[predicted_output,expected_output,confidence_value]
            if(expected_output==predicted_output):
                test_accuracy+=1


        test_accuracy=float(test_accuracy)*100/len(testing_data)

        train_accuracy=0.0

        for instance in training_data:
            expected_output = features_values['Class'][int(instance[-1])]
            predicted_output = features_values['Class'][int(neural_network.predictor(instance[:-1]))]
            confidence_value = neural_network.activation(instance[:-1])
            instance_output_mapping[str(instance)] = [predicted_output,expected_output,confidence_value]
            if (expected_output == predicted_output):
                train_accuracy += 1

        train_accuracy = float(train_accuracy)*100/len(training_data)
        avg_test_accuracy += test_accuracy
        avg_train_accuracy += train_accuracy

    avg_train_accuracy = avg_train_accuracy/(1.0*n)
    avg_test_accuracy = avg_test_accuracy/(1.0*n)

    return instance_output_mapping


def roc_plotter(instance_confidence_list):
        instance_confidence_list = instance_confidence_list[:]
        instance_confidence_list.sort(key = lambda x : -x[0]) # [(c1,y1),......]
        instances_list = []
        confidence_list = []
        for tuple_instance in instance_confidence_list:
                confidence_list.append(tuple_instance[0])
                instances_list.append(tuple_instance[1])

        num_neg = instances_list.count(0.0)
        num_pos = instances_list.count(1.0)
        TP = 0
        FP = 0
        last_TP = 0
        coordinate_points = [(0.0,0.0)]
        #print instances_list[0]
        if instances_list[0] == 1.0:
                TP += 1
        else:
                FP += 1
        for index in range(1,len(instances_list)):
                if (confidence_list[index] != confidence_list[index-1]) and (instances_list[index] == 0.0) and (TP > last_TP):
                        FPR = (FP*1.0)/num_neg
                        TPR = (TP*1.0)/num_pos
                        coordinate_points.append((FPR,TPR))
                        last_TP = TP
                if instances_list[index] == 1.0:
                        TP += 1
                else:
                        FP += 1
                #print (TP,FP)
        FPR = (FP*1.0)/num_neg
        TPR = (TP*1.0)/num_pos
        coordinate_points.append((FPR,TPR))
        for point in coordinate_points:
                print str(point[0])
        for point in coordinate_points:
                print str(point[1])




if __name__ == "__main__":

    #training file pulled from cmd
    tf = sys.argv[1]
    nfolds=sys.argv[2]
    lr=sys.argv[3]
    #nepochs=sys.argv[4]


    #file I/O
    #open the file in read mode
    tfp=open(tf,"r")
    train_data, train_meta = loadarff(tfp)

    attributes = []
    table = [list(i) for i in train_data]
    att_names = train_meta.names()
    for i in att_names:
        attributes.append(list(train_meta.__getitem__(i)[1]))


    ordered_features=att_names
    features_values=attributes
    features_values=dict(zip(ordered_features,features_values))
    training_data=table
    #opclass=train_data[163][-1]
    #print len(train_data)
    #print sigmo(train_data[1][1])

    #print opclass


