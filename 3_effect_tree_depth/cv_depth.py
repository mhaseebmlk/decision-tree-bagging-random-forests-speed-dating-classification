import sys
import time
import math 
import statistics
from collections import Counter

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

__TIME_DEBUG = False

TRAINING_DATA_FILE_NAME = 'trainingSet.csv'
# TRAINING_DATA_FILE_NAME = 'trainingSetSmall.csv'
TRAINING_DATA_PATH = '../1_preprocessing/' + TRAINING_DATA_FILE_NAME

RANDOM_STATE 	= 18
FRAC 			= 1

RANDOM_STATE_2 	= 32
FRAC_2 			= 0.5

###################### Decision Tree functions ################################
class InternalTreeNode:
	def __init__(self, attribute_):
		self.attribute = attribute_ # the attribute that this internal node holds
		self.children = None

class LeafTreeNode:
	def __init__(self,label_):
		self.label = label_

def build_tree(examples, algorithm, depth=0, max_depth=8):
	if len(examples) == 0:
		print ('===== [build_tree] len(examples) == 0')
		return None

	# if all examples have the same label, return leaf node with the label
	examples_labels = sorted(examples['decision'].unique())
	if len(examples_labels) == 1:
		label = examples_labels[0]
		return LeafTreeNode(label)

	# num_attributes cond: if all attributes have been used i.e. only 'decision' attribute left in df, return leaf node with majority label
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

		# if no more examples with this value, add an leaf node for this child of root
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
		attributes = np.random.choice(attributes, k, replace=False)
	for attribute in attributes:
		gg = gini_gain(examples, attribute)
		if gg > best_gg:
			best_gg = gg
			best_attribute_ = attribute
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
		child_node = root.children[row_attribute_val] # TODO: this may throw index out of range error if all values of an attribute were not present in the traiining set
		return predict_row_DT(row, child_node)
	raise Exception('Unknown node type received for root node: {}'.format(type(root), root))
###############################################################################

###################### Bagging functions ######################################
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

def k_fold_cross_validation(data, K, fold_sz, max_tree_depth, estimator):
	"""
	This version of k_fold_cross_validation is specific to the current task i.e. tuning the depth hyper parameter for all the trees
	"""

	M = 30 # number of trees to learn

	org_index = data.index.tolist()
	S, i = list(), 0
	for _ in range(K):
		S.append(data.loc[org_index[i:i+fold_sz]])
		i=i+fold_sz

	accuracies = list()
	for idx in range(K):
		test_set = S[idx]
		training_folds = [S[i] for i in range(K) if i != idx]
		training_set = pd.concat(training_folds)

		if estimator == 'DT':
			tree = build_tree(training_set, estimator, 0, max_tree_depth)
			accuracy = round(get_accuracy_DT(tree, test_set), 2)
			accuracies.append(accuracy)
		elif estimator == 'BT':
			training_samples = [training_set.sample(frac=1.0, replace=True) for _ in range(M)]
			trees = list(map(lambda training_sample: build_tree(training_sample, 'BT', 0, max_tree_depth), training_samples))

			accuracy = round(get_accuracy_BT(trees, test_set), 2)
			accuracies.append(accuracy)
		elif estimator == 'RF':
			training_samples = [training_set.sample(frac=1.0, replace=True) for _ in range(M)]
			trees = list(map(lambda training_sample: build_tree(training_sample, 'RF', 0, max_tree_depth), training_samples))

			accuracy = round(get_accuracy_BT(trees, test_set), 2)
			accuracies.append(accuracy)
		else:
			raise Exception('Unknown model: {}'.format(estimator))

	return accuracies

def main():
	t0 = time.time()

	df_training_org = pd.read_csv(TRAINING_DATA_PATH)
	df_training = df_training_org.sample(frac=FRAC, random_state=RANDOM_STATE).sample(frac=FRAC_2, random_state=RANDOM_STATE_2)

	# K-fold cross validation
	K = 10
	# K = 2
	FOLD_SZ_PERCENTAGE = 10
	TREE_DEPTHS = [3,5,7,9]
	# TREE_DEPTHS = [3,5,9]
	
	fold_sz = int(FOLD_SZ_PERCENTAGE/100 * len(df_training))

	model_accuracies = {'DT': dict(), 'BT': dict(), 'RF': dict()}
	# model_accuracies = {'RF': dict()}
	model_statistics = {'DT': list(), 'BT': list(), 'RF': list()}
	# model_statistics = {'RF': list()}

	for model in model_accuracies.keys():
		for d in TREE_DEPTHS:
			scores = k_fold_cross_validation(df_training, K, fold_sz, d, model)
			model_accuracies[model][d] = scores

	if __TIME_DEBUG:
		print ('model_accuracies = ', model_accuracies)

	# Statistics for each model
	for model in model_accuracies:
		for depth in model_accuracies[model].keys():
			accuracies = model_accuracies[model][depth]
			avg_accuracy = sum(accuracies) / len(accuracies) 
			sd = statistics.stdev(accuracies)
			standard_error = sd / math.sqrt(K)
			model_statistics[model].append((depth, avg_accuracy, standard_error))

	if __TIME_DEBUG:
		print ('model_statistics = ', model_statistics)

	# Graphing
	depths_DT = [t[0] for t in model_statistics['DT']]
	dt_avg_accuracies = [t[1] for t in model_statistics['DT']]
	dt_standard_errors = [t[2] for t in model_statistics['DT']]

	depths_BT = [t[0] for t in model_statistics['BT']]
	bt_avg_accuracies = [t[1] for t in model_statistics['BT']]
	bt_standard_errors = [t[2] for t in model_statistics['BT']]

	depths_RF = [t[0] for t in model_statistics['RF']]
	rf_avg_accuracies = [t[1] for t in model_statistics['RF']]
	rf_standard_errors = [t[2] for t in model_statistics['RF']]

	assert (depths_DT == depths_BT == depths_RF)
	depths = depths_DT
	# depths = depths_RF

	file_name = 'learning_curves.png'
	fig, ax = plt.subplots()
	ax.errorbar(depths, dt_avg_accuracies, yerr=dt_standard_errors, label='DT')
	ax.errorbar(depths, bt_avg_accuracies, yerr=bt_standard_errors, label='BT')
	ax.errorbar(depths, rf_avg_accuracies, yerr=rf_standard_errors, label='RF')
	ax.legend()
	title='Model Accuracy vs. Maximum Tree Depth'
	plt.xlabel('Maximum Tree Depth')
	plt.ylabel('Model Accuracy')
	plt.title(title)
	plt.savefig(file_name)

	if __TIME_DEBUG:
		t1 = time.time()
		print ('Time taken: ',t1-t0)

if __name__ == '__main__':
	main()
