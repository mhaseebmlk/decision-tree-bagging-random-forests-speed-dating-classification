import sys
import time
import math 

from collections import Counter

import pandas as pd
import numpy as np

if len(sys.argv) != 4:
	print('Usage: \n\tpython3 trees.py trainingSet.csv testSet.csv [1, 2, 3]')
	exit(1)

TRAINING_DATA_FILE_NAME = sys.argv[1]
# TRAINING_DATA_FILE_NAME = 'trainingSetSmall.csv'
TRAINING_DATA_PATH = '../1_preprocessing/' + TRAINING_DATA_FILE_NAME
TEST_DATA_FILE_NAME = sys.argv[2]
TEST_DATA_PATH = '../1_preprocessing/' + TEST_DATA_FILE_NAME

MODEL_IDX = sys.argv[3]

if MODEL_IDX not in {'1','2', '3'}:
	print('Model Index must be either 1, 2, or 3.')
	exit(1)

__DEBUG 		= False
__TIME_DEBUG 	= False

###################### Decision Tree functions ################################
class InternalTreeNode:
	def __init__(self, attribute_):
		self.attribute = attribute_ # the attribute that this internal node holds
		self.children = None

class LeafTreeNode:
	def __init__(self,label_):
		self.label = label_

def print_level_order(root):
    q, ret = [(None,root)], []
    while any(q):
        ret.append([(node[0], node[1].attribute) if isinstance(node[1], InternalTreeNode) else (node[0], node[1].label) for node in q])

        q = [(node[1].children.index(child),child) for node in q if isinstance(node[1],InternalTreeNode) for child in node[1].children if child]

    for level in ret:
   		print (level)

def print_level_order_2(root):
    if not root: print ([])
    ret = []
    stack = [(None,root)]
    while stack:
        temp = []
        next_stack = []
        for t in stack:
        	# print ('t = ',t )
        	if isinstance(t[1], InternalTreeNode):
        		temp.append( (t[0], t[1].attribute) )
        	elif isinstance(t[1], LeafTreeNode):
        		temp.append( (t[0], t[1].label) )
        	# else:
        	# 	raise Exception('Invalid tree node:', t[1])

        	if isinstance(t[1], InternalTreeNode):
        		for child in t[1].children:
        			next_stack.append( ( t[1].children.index(child), child) )
        	stack = next_stack

        ret.append(temp)

    for level in ret:
   		print (level)

def decisionTree(trainingSet, testSet):
	# train
	max_tree_depth = 8
	tree = build_tree(trainingSet, 'DT', 0, max_tree_depth)

	#print_level_order(tree)
	#print ('----')
	#print_level_order_2(tree)

	# test
	print ('Training Accuracy DT: {}'.format(round(get_accuracy_DT(tree, trainingSet), 2)))
	print ('Testing Accuracy DT: {}'.format(round(get_accuracy_DT(tree, testSet), 2)))

def build_tree(examples, algorithm, depth=0, max_depth=8):
	if len(examples) == 0:
		print ('===== [build_tree] len(examples) == 0')
		return None

	# if all examples have the same label, return leaf node with the label
	examples_labels = sorted(examples['decision'].unique())
	if len(examples_labels) == 1:
		label = examples_labels[0]
		return LeafTreeNode(label)

	# if all attributes have been used i.e. only 'decision' attribute left in df, return leaf node with majority label
	# depth condition: set the majority label if the depth has been reached
	# num examples condtion: stop growing when the number of examples in a node < 50
	if list(examples.columns) == ['decision'] or depth >= max_depth or len(examples)<50:
		label = majority_label(examples)
		return LeafTreeNode(label)

	# create a root/internal node with the best attribute from the given examples (having maximum gini gain)
	A = best_attribute(examples, algorithm)
	root = InternalTreeNode(A)

	attribute_vals = sorted(examples[A].unique())
	root.children = [None]*(len(attribute_vals)+1)
	for v in attribute_vals:
		val_filter = examples[A].isin([v])
		examples_v = examples[val_filter]

		# if no more examples with this value, add a leaf node for this child of root
		if len(examples_v) == 0:
			label = majority_label(examples)
			root.children[v] = LeafTreeNode(label)
		else:
			examples_v = examples_v.drop([A], axis=1)
			root.children[v] = build_tree(examples_v, algorithm, depth+1, max_depth)

	return root

def best_attribute(examples, algorithm):
	best_attribute_, best_gg = None, float('-inf')
	attributes = examples.drop(['decision'], axis=1).columns
	if algorithm in ['RF']:	
		k = math.ceil(math.sqrt(len(attributes)))
		attributes = np.random.choice(attributes, k, replace=False) # we need K distinct attributes
	for attribute in attributes:
		gg = gini_gain(examples, attribute)
		if gg > best_gg:
			best_gg = gg
			best_attribute_ = attribute
	print ('gini sample: {} column length: {}'.format(gini(examples), len(examples.columns)-1))
	print ('Max split attribute: {}, max gini gain: {}'.format(best_attribute_, best_gg))
	return best_attribute_

def gini_gain(examples, attribute):
	gini_examples = gini(examples)
	gini_atrbt_vals = 0.0
	attribute_vals = sorted(examples[attribute].unique())
	for v in attribute_vals:
		val_filter = examples[attribute].isin([v])
		examples_v = examples[val_filter]
		gini_examples_v = gini(examples_v)
		gini_atrbt_vals += (len(examples_v)/len(examples))*gini_examples_v
	return gini_examples - gini_atrbt_vals

def gini(examples):
	all_labels = examples['decision'].unique()
	sum_ = 0.0
	for l in all_labels:
		filter_ = examples['decision'].isin([l])
		filtered_examples =  examples[filter_]
		prob = len(filtered_examples) / len(examples)
		sum_ += pow(prob, 2)
	return 1 - sum_

def majority_label(examples):
	return examples['decision'].value_counts(ascending=True).index[-1]

def get_accuracy_DT(tree,examples):
	predictions = examples.apply(predict_DT, axis=1, args=(tree,))
	prediction_counts = predictions.value_counts().values.tolist()
	num_correct_predictions = prediction_counts[0]
	accuracy = num_correct_predictions/(sum(prediction_counts))
	return accuracy

def predict_DT(row, tree):
	predicted_label = predict_row_DT(row, tree)
	return 1 if row['decision'] == predicted_label else -1

def predict_row_DT(row, root):
	if isinstance(root, LeafTreeNode):
		return root.label
	if isinstance(root, InternalTreeNode):
		root_attribute = root.attribute
		row_attribute_val = row[root_attribute]
		child_node = root.children[row_attribute_val]
		return predict_row_DT(row, child_node)
	raise Exception('Unknown node type received for root node: {}'.format(type(root), root))
###############################################################################

###################### Bagging functions ######################################
def bagging(trainingSet, testSet):
	# train
	M = 30
	t3 = time.time()
	
	training_samples = [trainingSet.sample(frac=1.0, replace=True) for _ in range(M)]
	trees = list(map(lambda training_sample: build_tree(training_sample, 'BT', 0, 8), training_samples))
	
	t4 = time.time()
	if __TIME_DEBUG:
		print ('Training took time: ',t4-t3)

	# test
	t5 = time.time()
	
	print ('Training Accuracy BT: {}'.format(round(get_accuracy_BT(trees, trainingSet), 2)))
	print ('Testing Accuracy BT: {}'.format(round(get_accuracy_BT(trees, testSet), 2)))
	
	t6 = time.time()
	if __TIME_DEBUG:
		print ('Computing Accuracies took time: ',t6-t5)

def get_accuracy_BT(trees,examples):
	correct_incorrect_classifications = list()
	for index, row in examples.iterrows():
		predicted_labels = map(lambda tree: predict_row_DT(row, tree), trees)
		predicted_label = most_frequent(predicted_labels)
		correct_incorrect_classifications.append(1) if row['decision'] == predicted_label else correct_incorrect_classifications.append(-1)
	num_correct_predictions = correct_incorrect_classifications.count(1)
	return num_correct_predictions / len(correct_incorrect_classifications)

def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0] 
###############################################################################

###################### Random Forests functions ###############################
def randomForests(trainingSet, testSet):
	# train
	M = 30
	t3 = time.time()
	
	training_samples = [trainingSet.sample(frac=1.0, replace=True) for _ in range(M)]
	trees = list(map(lambda training_sample: build_tree(training_sample, 'RF', 0, 8), training_samples))
	
	t4 = time.time()
	if __TIME_DEBUG:
		print ('Training took time: ',t4-t3)

	# test
	t5 = time.time()
	
	print ('Training Accuracy RF: {}'.format(round(get_accuracy_BT(trees, trainingSet), 2)))
	print ('Testing Accuracy RF: {}'.format(round(get_accuracy_BT(trees, testSet), 2)))
	
	t6 = time.time()
	if __TIME_DEBUG:
		print ('Computing Accuracies took time: ',t6-t5)
###############################################################################

def main():
	df_training = pd.read_csv(TRAINING_DATA_PATH)
	df_test = pd.read_csv(TEST_DATA_PATH)

	t0 = time.time()

	if MODEL_IDX == '1':
		decisionTree(df_training, df_test)
	elif MODEL_IDX == '2':
		bagging(df_training, df_test)
	else:
		randomForests(df_training, df_test)

	t1 = time.time()	
	if __TIME_DEBUG:
		print ('Time taken: ',t1-t0)

if __name__ == '__main__':
	main()
