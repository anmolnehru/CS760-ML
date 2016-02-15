#Anmol Mohanty
#CS760 ML code
#Assignment 2 - Naive Bayes vs TAN comparative study
#Submission date : Valentines day :P. 2/14/16
import math
import sys

class NaiveBayes:

	def __init__(self,ordered_features,features_values,training_data):
		self.ordered_features = ordered_features #Contains features as in file (without class)
		self.features_values = features_values #Dictionary (key,value) --> Key:Attribute Value:PossibleValues (includes class)
		self.training_data =training_data
		self.individualFeatureCounts = {} #
		for feature in self.ordered_features:
			self.individualFeatureCounts[feature] = {}
			possible_feature_values = features_values[feature]
			for possible_feature_value in possible_feature_values:
				self.individualFeatureCounts[feature][possible_feature_value] = {features_values['class'][0] : 0,features_values['class'][1] : 0}
		self.classCounts = {}
		for class_value in features_values['class']:
			self.classCounts[class_value] = 0


	def initializeCounters(self):
		for training_instance in self.training_data:
			expected_class = training_instance[-1]
			self.classCounts[expected_class] += 1
			for feature in self.ordered_features:
				feature_index = self.ordered_features.index(feature)
				feature_value = training_instance[feature_index]
				self.individualFeatureCounts[feature][feature_value][expected_class] += 1

	def probability_class(self, class_value):
		#laplace estimates
		probability = ((self.classCounts[class_value]+1)*1.0)/(len(self.training_data)+len(self.classCounts.keys()))
		return probability

	def probability_xi_class(self,attribute,attribute_value,class_value):
		probability = ((self.individualFeatureCounts[attribute][attribute_value][class_value] + 1)*1.0)/(self.classCounts[class_value]+len(self.features_values[attribute]))
		return probability

	def classify(self,test_instance):
		class_value_1 = features_values['class'][0]
		probability_class_value_1 = self.probability_class(class_value_1)
		product_probability_class_value_1 = 1.0
		class_value_2 = features_values['class'][1]
		probability_class_value_2 = self.probability_class(class_value_2)
		product_probability_class_value_2 = 1.0
		for feature in self.ordered_features:
			feature_index = self.ordered_features.index(feature)
			features_value = test_instance[feature_index]
			product_probability_class_value_1 *= self.probability_xi_class(feature,features_value,class_value_1)
			product_probability_class_value_2 *= self.probability_xi_class(feature,features_value,class_value_2)

		probability_class_value_1 = probability_class_value_1 * product_probability_class_value_1
		probability_class_value_2 = probability_class_value_2 * product_probability_class_value_2

		if (probability_class_value_1 > probability_class_value_2):
			posterior_probability = probability_class_value_1/(probability_class_value_1+probability_class_value_2)
			predicted_class = class_value_1
		else:
			posterior_probability = probability_class_value_2/(probability_class_value_1+probability_class_value_2)
			predicted_class = class_value_2

		return (predicted_class,posterior_probability)

	def test_looper(self,testing_data):
		correctly_classified_count = 0
		for feature in ordered_features:
			print feature+" "+"class"
		print("")
		for instance in testing_data:
			predicted_class,posterior_probability = self.classify(instance)
			actual_class = instance[-1]
			if (predicted_class == actual_class):
				correctly_classified_count += 1
            #print predicted_class+" "+actual_class+" "+"{0:0.f}".format(posterior_probability)
			print predicted_class+" "+actual_class+" "+"{0:.12f}".format(posterior_probability)
		print("\n"+str(correctly_classified_count))

class TAN:

	def __init__(self,ordered_features,features_values,training_data):
		self.ordered_features = ordered_features #Contains features as in file (without class)
		self.features_values = features_values #Dictionary (key,value) --> Key:Attribute Value:PossibleValues (includes class)
		self.training_data =training_data
		self.individualFeatureCounts = {} #
		for feature in self.ordered_features:
			self.individualFeatureCounts[feature] = {}
			possible_feature_values = features_values[feature]
			for possible_feature_value in possible_feature_values:
				self.individualFeatureCounts[feature][possible_feature_value] = {features_values['class'][0] : 0,features_values['class'][1] : 0}
		self.jointFeatureCounts = {} #
		for feature_i in self.ordered_features:
			self.jointFeatureCounts[feature_i] = {}
			possible_feature_i_values = features_values[feature_i]
			for feature_j in self.ordered_features:
				self.jointFeatureCounts[feature_i][feature_j]={}
				possible_feature_j_values = features_values[feature_j]
				for feature_i_value in possible_feature_i_values:
					self.jointFeatureCounts[feature_i][feature_j][feature_i_value] = {}
					for feature_j_value in possible_feature_j_values:
						self.jointFeatureCounts[feature_i][feature_j][feature_i_value][feature_j_value] = {features_values['class'][0] : 0,features_values['class'][1] : 0}
		self.classCounts = {}
		for class_value in features_values['class']:
			self.classCounts[class_value] = 0
		self.MutualInformation = []
		self.parents = {}
		for feature in self.ordered_features:
			self.MutualInformation.append([-1]*len(self.ordered_features))
			self.parents[feature] = ['class']


	def initializeCounters(self):
		for training_instance in self.training_data:
			expected_class = training_instance[-1]
#        print "expected_class"
			self.classCounts[expected_class] += 1
			for feature_i in self.ordered_features:
				feature_i_index = self.ordered_features.index(feature_i)
				feature_i_value = training_instance[feature_i_index]
				self.individualFeatureCounts[feature_i][feature_i_value][expected_class] += 1
				for feature_j in self.ordered_features:
					feature_j_index = self.ordered_features.index(feature_j)
					feature_j_value = training_instance[feature_j_index]
					self.jointFeatureCounts[feature_i][feature_j][feature_i_value][feature_j_value][expected_class] += 1

	def calculateMutualInformation(self):

		for feature_i_index in range(0,len(self.ordered_features)):
			feature_i = self.ordered_features[feature_i_index]
			for feature_j_index in range(feature_i_index,len(self.ordered_features)):
				feature_j = self.ordered_features[feature_j_index]
				if(feature_i_index != feature_j_index):
					mutualInformation = 0.0
					for feature_i_value_index in range(0,len(self.features_values[feature_i])):
						feature_i_value = self.features_values[feature_i][feature_i_value_index]
						for feature_j_value_index in range(0,len(self.features_values[feature_j])):
							feature_j_value = self.features_values[feature_j][feature_j_value_index]
							for class_value in self.classCounts.keys():
								joint_probability_term = ((self.jointFeatureCounts[feature_i][feature_j][feature_i_value][feature_j_value][class_value]+1)*1.0)/(len(self.training_data)+(len(self.features_values[feature_i])*len(self.features_values[feature_j])*len(self.classCounts.keys())))
								probability_x_ij_y = ((self.jointFeatureCounts[feature_i][feature_j][feature_i_value][feature_j_value][class_value]+1)*1.0)/(self.classCounts[class_value]+(len(self.features_values[feature_i])*len(self.features_values[feature_j])))
								probability_x_i_y = ((self.individualFeatureCounts[feature_i][feature_i_value][class_value]+1)*1.0)/(self.classCounts[class_value]+len(self.features_values[feature_i]))
								probability_x_j_y = ((self.individualFeatureCounts[feature_j][feature_j_value][class_value]+1)*1.0)/(self.classCounts[class_value]+len(self.features_values[feature_j]))
								log_term = (probability_x_ij_y/(probability_x_i_y*probability_x_j_y))
								log_term = math.log(log_term,2)
								mutualInformation += (joint_probability_term * log_term)
					self.MutualInformation[feature_i_index][feature_j_index] = mutualInformation
					self.MutualInformation[feature_j_index][feature_i_index] = mutualInformation

	def probability_class(self, class_value):
		probability = ((self.classCounts[class_value]+1)*1.0)/(len(self.training_data)+len(self.classCounts.keys()))
		return probability

	def probability_xi_class(self,attribute,attribute_value,class_value):
		probability = ((self.individualFeatureCounts[attribute][attribute_value][class_value] + 1)*1.0)/(self.classCounts[class_value]+len(self.features_values[attribute]))
		return probability

	def probability_xi_xj_class(self,attribute,parent_attribute,attribute_value,parent_attribute_value,class_value):
		probability = ((self.jointFeatureCounts[attribute][parent_attribute][attribute_value][parent_attribute_value][class_value] + 1)*1.0)/(self.individualFeatureCounts[parent_attribute][parent_attribute_value][class_value]+len(self.features_values[attribute]))
		return probability


	def classify(self,test_instance):
		class_value_1 = features_values['class'][0]
		probability_class_value_1 = self.probability_class(class_value_1)
		product_probability_class_value_1 = 1.0
		class_value_2 = features_values['class'][1]
		probability_class_value_2 = self.probability_class(class_value_2)
		product_probability_class_value_2 = 1.0
		for feature in self.ordered_features:
			feature_index = self.ordered_features.index(feature)
			features_value = test_instance[feature_index]
			if (len(self.parents[feature]) == 1):
				product_probability_class_value_1 *= self.probability_xi_class(feature,features_value,class_value_1)
				product_probability_class_value_2 *= self.probability_xi_class(feature,features_value,class_value_2)
			else:
				parent_feature = self.parents[feature][0]
				parent_feature_index = self.ordered_features.index(parent_feature)
				parent_feature_value = test_instance[parent_feature_index]
				product_probability_class_value_1 *= self.probability_xi_xj_class(feature,parent_feature,features_value,parent_feature_value,class_value_1)
				product_probability_class_value_2 *= self.probability_xi_xj_class(feature,parent_feature,features_value,parent_feature_value,class_value_2)


		probability_class_value_1 = probability_class_value_1 * product_probability_class_value_1
		probability_class_value_2 = probability_class_value_2 * product_probability_class_value_2
		if (probability_class_value_1 > probability_class_value_2):
			posterior_probability = probability_class_value_1/(probability_class_value_1+probability_class_value_2)
			predicted_class = class_value_1
		else:
			posterior_probability = probability_class_value_2/(probability_class_value_1+probability_class_value_2)
			predicted_class = class_value_2

		return (predicted_class,posterior_probability)

	def test_looper(self,testing_data):
		correctly_classified_count = 0
		for feature in ordered_features:
			output_str = feature+" "
			for parent in self.parents[feature]:
				output_str += parent+" "
			print output_str
		print("")
		for instance in testing_data:
			predicted_class,posterior_probability = self.classify(instance)
			actual_class = instance[-1]
			if (predicted_class == actual_class):
				correctly_classified_count += 1
            #print predicted_class+" "+actual_class+" "+"{0:.10f}".format(posterior_probability)
			print predicted_class+" "+actual_class+" "+"{0:.12f}".format(posterior_probability)
		print("\n"+str(correctly_classified_count))

	def getSpanningTree(self):
		maximum_spanning_tree = self.MST(range(len(self.ordered_features)))

		for edge in maximum_spanning_tree:
			src_vertex = self.ordered_features[edge[0]]
			dst_vertex = self.ordered_features[edge[1]]
			self.parents[dst_vertex].insert(0,src_vertex)

		'''
		for feature in self.ordered_features:
			print feature+" "+str(self.parents[feature])
		'''

	def comparator(self,edge_1,edge_2):
		mutualInformation_edge_1 = self.MutualInformation[edge_1[0]][edge_1[1]]
		mutualInformation_edge_2 = self.MutualInformation[edge_2[0]][edge_2[1]]
		if (mutualInformation_edge_1 > mutualInformation_edge_2):
			return -1
		elif (mutualInformation_edge_2 > mutualInformation_edge_1):
			return 1
		elif (edge_1[0] < edge_2[0]):
			return -1
		elif (edge_1[0] > edge_2[0]):
			return 1
		elif (edge_1[1] < edge_2[1]):
			return -1
		else:
			return 1

	def MST(self,vertices):
		selected_vertices = set()
		selected_edges = set()
		selected_vertices.add(0) #Taking the vertex which first occurs in the file
		while len(selected_vertices) != len(vertices):
			candidate_edges = set()
			for selected_vertex in selected_vertices:
				for generic_vertex in vertices:
					if (generic_vertex not in selected_vertices and self.MutualInformation[selected_vertex][generic_vertex] != -1):
						candidate_edges.add((selected_vertex,generic_vertex))

			correct_edge = sorted(candidate_edges, cmp = self.comparator)[0]
			selected_edges.add(correct_edge)
			selected_vertices.add(correct_edge[1])

		return selected_edges

def read_file(filename,train_flag = False):
	fp = open(filename,"r")
	lines = fp.read().split("\n")
	lines = lines[1:] #removing the relation
	lines = [line for line in lines if line!='']
	data = []
	if(train_flag):
		ordered_features = [] #store names of features
		features_values = {}  #store the possible values they can take
		for line in lines:
			if line.startswith("@attribute"):
				line_parts =line.split(' ')
				ordered_features.append(eval(line_parts[1]))
				if len(line_parts) <4:
					features_values[eval(line_parts[1])] = line_parts[-1]
				else:
					features_values[eval(line_parts[1])] = []  ##fix and look into this logic
					for index in range(3,len(line_parts)):
						if (len(line_parts[index]) > 1):
							features_values[eval(line_parts[1])].append(line_parts[index][:-1])
			elif not line.startswith("@") and not line.startswith("%"): #actual data
				data.append(line.split(','))
		ordered_features.remove('class') #not a part of attribute feature space
		#aggregate the return data type into one variable
		return_data = (ordered_features,features_values,data)
	else:	#testing data, just need the csv's
		for line in lines:
			if not line.startswith("@") and not line.startswith("%"):
				data.append(line.split(','))
		return_data = data

	return return_data

def callNB():
	myNB = NaiveBayes(ordered_features,features_values,training_data)
	myNB.initializeCounters()
	myNB.test_looper(testing_data)

def callTAN():
	myTAN = TAN(ordered_features,features_values,training_data)
	myTAN.initializeCounters()
	myTAN.calculateMutualInformation()
	myTAN.getSpanningTree()
	myTAN.test_looper(testing_data)


#Main portion of the code

if __name__ == "__main__" :

	#first parameter :: train set
	training_filename = sys.argv[1]
	#second parameter :: test set
	testing_filename = sys.argv[2]
	#third the type of model being chosen
	model = sys.argv[3]

	#populating the training and testing space
	ordered_features,features_values,training_data = read_file(training_filename,True)
	testing_data = read_file(testing_filename,False)

	if model == 't':
		callTAN()
	else:
		callNB()
