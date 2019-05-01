""" 
	This script performs hypothesis testing between two models: DT and BT
	as the depth of the tree increases.
"""
from scipy import stats
SIGNIFICANCE_LEVEL  = 0.05

H0 = 'As the tree depth increases, the mean accuracies for both DT and BT remain the same i.e. their performance does not change with respect to each other.'
H1 = 'As the tree depth increases, the mean accuracies of DT != mean accuracies of BT i.e. their performance differs with respect to each other.'

print ('H0: {}'.format(H0))
print ('H1: {}'.format(H1))

# Accuracies from the 10-Fold Cross Validation for each depth and each model
dt_accs_depth_3 = [0.72, 0.77, 0.74, 0.69, 0.79, 0.75, 0.75, 0.76, 0.73, 0.7]
dt_accs_depth_5 = [0.75, 0.75, 0.72, 0.7, 0.78, 0.75, 0.72, 0.76, 0.75, 0.7]
dt_accs_depth_7 = [0.75, 0.73, 0.71, 0.68, 0.74, 0.75, 0.72, 0.78, 0.77, 0.7]
dt_accs_depth_9 = [0.76, 0.73, 0.71, 0.67, 0.73, 0.72, 0.71, 0.8, 0.74, 0.72]
dt_accs = [dt_accs_depth_3, dt_accs_depth_5, dt_accs_depth_7, dt_accs_depth_9]

bt_accs_depth_3 = [0.73, 0.77, 0.73, 0.7, 0.79, 0.75, 0.75, 0.76, 0.73, 0.7]
bt_accs_depth_5 = [0.74, 0.78, 0.72, 0.7, 0.78, 0.75, 0.75, 0.77, 0.73, 0.7]
bt_accs_depth_7 = [0.76, 0.78, 0.74, 0.72, 0.79, 0.75, 0.74, 0.77, 0.76, 0.7]
bt_accs_depth_9 = [0.75, 0.77, 0.75, 0.72, 0.76, 0.75, 0.75, 0.8, 0.76, 0.71]
bt_accs = [bt_accs_depth_3, bt_accs_depth_5, bt_accs_depth_7, bt_accs_depth_9]

depths = [3,5,7,9]
for i in range(len(depths)):
	depth = depths[i]
	t_val, p_val = stats.ttest_rel(dt_accs[i], bt_accs[i])
	print ('Depth: {} H0 for DT and BT: t-statistics = {}, p-value = {} Reject with significance level of {}? {}'.format(depth, t_val, p_val, SIGNIFICANCE_LEVEL, (p_val < SIGNIFICANCE_LEVEL)))
