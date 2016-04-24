import math
import sys
import random

class NeuralNetwork:

	def __init__(self,input_units_count):
		self.weights = []
		for index in range(0,input_units_count):
			self.weights.append(0.1)
		self.bias = 0.1

	def sigmoid(self,x):
		return 1.0/(1.0 + math.exp(-x))

	def dot(self,x,y):
		dot_product = 0.0
		for index in range(0,len(x)):
			dot_product += float(x[index] * y[index])
		return dot_product

	def activation(self,x):
		return self.sigmoid(self.dot(x,self.weights)+self.bias)

	def thresholder(self,activation_value):
		if activation_value > 0.5:
			return 1.0
		else:
			return 0.0

	def predictor(self,test_instance):
		return self.thresholder(self.activation(test_instance))

	def trainer(self,training_set, learning_rate, epoch_count):

		for current_epoch in range(0,epoch_count):
			for training_instance in training_set:
				# 1. Find O/P of this training_instance
				# 2. Find the error derivative for each weight
				# 3. Update the weights
				activation_value = self.activation(training_instance[:-1])
				error_derivates = []
				expected_output = training_instance[-1]
				sigmoid_derivative_value = activation_value * (1 - activation_value)
				self.bias += learning_rate * (expected_output - activation_value) * sigmoid_derivative_value * 1.0
				for index in range(0,len(self.weights)):
					x_i = training_instance[index]
					error_derivates.append((expected_output - activation_value) * sigmoid_derivative_value * x_i)
					self.weights[index] += learning_rate*error_derivates[index]
				'''
				for index in range(0,len(self.weights)):
					self.weights[index] += learning_rate*error_derivates[index]
				'''

def read_file(filename,train_flag = False):
	fp = open(filename,"r")
	lines = fp.read().split("\n")
	lines = lines[1:] #removing the relation
	lines = [line for line in lines if line!='']
	data = []
	if(train_flag):
		ordered_features = []
		features_values = {}
		for line in lines:
			if line.startswith("@attribute"):
				line_parts =line.split(' ')
				ordered_features.append(eval(line_parts[1]))
				if len(line_parts) <4:
					features_values[eval(line_parts[1])] = line_parts[-1]
				else:
					features_values[eval(line_parts[1])] = []
					for index in range(3,len(line_parts)):
						features_values[eval(line_parts[1])].append(line_parts[index][:-1])
			elif not line.startswith("@"):
				data.append(line.split(','))
		ordered_features.remove('Class')
		return_data = (ordered_features,features_values,data)
	else:
		for line in lines:
			if not line.startswith("@"):
				data.append(line.split(','))
		return_data = data

	return return_data


def straified_data_partitioner(training_data,n):
	training_data = training_data[:]
	partitions = []
	positive_training_instances = []
	negative_training_instances = []
	instance_fold_mapping = {}

	#Stratifying the instances
	for training_instance in training_data:
		if training_instance[-1] == 1:
			positive_training_instances.append(training_instance)
		else:
			negative_training_instances.append(training_instance)

	#print "Total +ve count : "+str(len(positive_training_instances))
	#print "Total -ve count : "+str(len(negative_training_instances))
	#Do the partitioning
	postive_training_instances_avg_count = len(positive_training_instances)/n
	negative_training_instances_avg_count = len(negative_training_instances)/n

	for partition_count in range(0,n):
		# 1. Get the average number of +ve instances
		# 2. Remove them from the positive_training_instances
		# 3. Insert in the current partition:
		random_positive_instances = random.sample(positive_training_instances,postive_training_instances_avg_count)
		for instance in random_positive_instances:
			instance_fold_mapping[str(instance)] = partition_count + 1
			positive_training_instances.remove(instance)
		partitions.append(random_positive_instances)

	partition_count = 0

	while len(positive_training_instances) != 0:
		# 1. Add one by one to the remaining paritions until the list goes empty
		partitions[partition_count].append(positive_training_instances[0])
		instance_fold_mapping[str(positive_training_instances[0])] = partition_count + 1
		positive_training_instances = positive_training_instances[1:]
		partition_count = (partition_count+1)% n

	for partition_count in range(0,n):
		# 1. Get the average number of -ve instances
		# 2. Remove them from the negative_training_instances
		# 3. Insert in the current partition:
		random_positive_instances = random.sample(negative_training_instances,negative_training_instances_avg_count)
		for instance in random_positive_instances:
			instance_fold_mapping[str(instance)] = partition_count + 1
			negative_training_instances.remove(instance)
			partitions[partition_count].append(instance)

	partition_count = 0

	while len(negative_training_instances) != 0:
		# 1. Add one by one to the remaining paritions until the list goes empty
		partitions[partition_count].append(negative_training_instances[0])
		instance_fold_mapping[str(negative_training_instances[0])] = partition_count + 1
		negative_training_instances = negative_training_instances[1:]
		partition_count = (partition_count+1) % n

	return partitions, instance_fold_mapping

def stratified_cross_validator(partitions_list,l,e):
	avg_train_accuracy = 0.0
	avg_test_accuracy = 0.0
	instance_output_mapping = {} # key : an instance , value : [predicted O/P, expected O/P, confidence]
	n = len(partitions_list)

	for iteration_count in range(0,n):
		training_partitions = []
		for i in range(iteration_count,iteration_count+n-1):
			if((i+1) > n):
				partition_number = (i+1)%n
			else:
				partition_number = (i+1)
			training_partitions.append(partition_number-1)
		valid_partition_numbers = list(range(0,n))
		for number in training_partitions:
			if number in training_partitions:
				valid_partition_numbers.remove(number)

		test_partition_number = valid_partition_numbers[0]
		training_data = []
		for partition_number in training_partitions:
			for instance in partitions_list[partition_number]:
				training_data.append(instance)

		testing_data = partitions_list[test_partition_number]
		if (n==1):
			training_data = testing_data

		neural_network = NeuralNetwork(len(ordered_features))
		random.shuffle(training_data)
		neural_network.trainer(training_data,l,e)

		test_accuracy = 0.0

		for instance in testing_data:
			expected_output = features_values['Class'][int(instance[-1])]
			predicted_output = features_values['Class'][int(neural_network.predictor(instance[:-1]))]
			confidence_value = neural_network.activation(instance[:-1])
			instance_output_mapping[str(instance)] = [predicted_output,expected_output,confidence_value]
			if (expected_output == predicted_output):
				test_accuracy += 1

		test_accuracy = float(test_accuracy)*100/len(testing_data)

		train_accuracy = 0.0

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
        #print "avg_test_accuracy="+str(avg_test_accuracy)
	return instance_output_mapping

def roc_plotter(instance_confidence_list):
	#print instance_confidence_list
        instance_confidence_list = instance_confidence_list[:]
	instance_confidence_list.sort(key = lambda x : -x[0]) # [(c1,y1),......]
	instances_list = []
	confidence_list = []
	for tuple_instance in instance_confidence_list:
		confidence_list.append(tuple_instance[0])
		instances_list.append(tuple_instance[1])

	num_neg = instances_list.count("Rock")
        #print "num_neg"+str(num_neg)
	num_pos = instances_list.count("Mine")
        #print "num_pos"+str(num_pos)
	TP = 0
	FP = 0
	last_TP = 0
	coordinate_points = []
	#print instances_list[0]
	if instances_list[0] == "Mine":
		TP += 1
	else:
		FP += 1
	for index in range(1,len(instances_list)):
		if (confidence_list[index] != confidence_list[index-1]) and (instances_list[index] == "Rock") and (TP > last_TP):
			FPR = (FP*1.0)/num_neg
			TPR = (TP*1.0)/num_pos
			coordinate_points.append([FPR,TPR])
                        #print FPR+" "+TPR
			last_TP = TP
		if instances_list[index] == "Mine":
			TP += 1
		else:
			FP += 1
		#print (TP,FP)
	FPR = (FP*1.0)/(num_neg)
	TPR = (TP*1.0)/(num_pos)
	coordinate_points.append([FPR,TPR])

        #print coordinate_points

        #for point in coordinate_points:
	#	print str(point[0])
	#for point in coordinate_points:
	#	print str(point[1])


def threshold(activation_value):
	if activation_value > 0.5:
		return 1.0
	else:
		return 0.0


if __name__ == "__main__" :

	from datetime import datetime
	startTime = datetime.now()

	training_filename = str(sys.argv[1])
	n = int(sys.argv[2])
	l = float(sys.argv[3])
	e = int(sys.argv[4])
	ordered_features,features_values,training_data = read_file(training_filename,True)
	positive_class = features_values['Class'][1]
	negative_class = features_values['Class'][0]

	for training_instance in training_data:
		training_instance[-1] = features_values['Class'].index(training_instance[-1])
		for index in range(0,len(training_instance)):
			training_instance[index] = float(training_instance[index])

	partitions, instance_fold_mapping = straified_data_partitioner(training_data,n)
	instance_output_mapping = stratified_cross_validator(partitions,l,e)

#mohanty
        instance_confidence_list=[]
	for index in range(0,len(training_data)):
		print str(instance_fold_mapping[str(training_data[index])])+"\t"+str(instance_output_mapping[str(training_data[index])][0])+"\t"+str(instance_output_mapping[str(training_data[index])][1])+"\t"+str(instance_output_mapping[str(training_data[index])][2])
                instance_confidence_list.append([instance_output_mapping[str(training_data[index])][2],str(instance_output_mapping[str(training_data[index])][1])])

        roc_plotter(instance_confidence_list)
