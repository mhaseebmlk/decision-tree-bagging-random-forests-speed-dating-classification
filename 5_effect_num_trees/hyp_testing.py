""" 
	This script performs hypothesis testing between two models: RF and BT
	as the number of trees increases.
"""
from scipy import stats
SIGNIFICANCE_LEVEL  = 0.05

H0 = 'As the number of trees increases, the mean accuracies for both RF and BT remain the same i.e. their performance does not change with respect to each other.'
H1 = 'As the number of trees increases, the mean accuracies of RF != mean accuracies of BT i.e. their performance changes with respect to each other.'

print ('H0: {}'.format(H0))
print ('H1: {}'.format(H1))

# Accuracies from the 10-Fold Cross Validation for each value of num_trees and each model
rf_accs_num_trees_10 = [0.75, 0.76, 0.73, 0.72, 0.74, 0.77, 0.73, 0.75, 0.75, 0.7]
rf_accs_num_trees_20 = [0.75, 0.77, 0.75, 0.69, 0.78, 0.78, 0.74, 0.77, 0.73, 0.73]
rf_accs_num_trees_40 = [0.75, 0.76, 0.73, 0.72, 0.78, 0.77, 0.74, 0.78, 0.73, 0.73]
rf_accs_num_trees_50 = [0.75, 0.77, 0.72, 0.71, 0.76, 0.77, 0.75, 0.77, 0.75, 0.72]
rf_accs = [rf_accs_num_trees_10, rf_accs_num_trees_20, rf_accs_num_trees_40, rf_accs_num_trees_50]

bt_accs_num_trees_10 = [0.76, 0.77, 0.73, 0.7, 0.71, 0.73, 0.72, 0.78, 0.72, 0.71]
bt_accs_num_trees_20 = [0.75, 0.76, 0.75, 0.72, 0.78, 0.72, 0.74, 0.8, 0.75, 0.7]
bt_accs_num_trees_40 = [0.79, 0.75, 0.73, 0.7, 0.77, 0.73, 0.73, 0.78, 0.75, 0.72]
bt_accs_num_trees_50 = [0.75, 0.75, 0.73, 0.7, 0.78, 0.74, 0.74, 0.79, 0.78, 0.71]
bt_accs = [bt_accs_num_trees_10, bt_accs_num_trees_20, bt_accs_num_trees_40, bt_accs_num_trees_50]

num_trees = [10,20,40,50]
for i in range(len(num_trees)):
	n_trees = num_trees[i]
	t_val, p_val = stats.ttest_rel(rf_accs[i], bt_accs[i])
	print ('Number of Trees: {} H0 for DT and BT: t-statistics = {}, p-value = {} Reject with significance level of {}? {}'.format(n_trees, t_val, p_val, SIGNIFICANCE_LEVEL, (p_val < SIGNIFICANCE_LEVEL)))
