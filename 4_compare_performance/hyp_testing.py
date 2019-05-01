""" 
	This script performs hypothesis testing between two models: DT and RF
	as the training fraction increases.
"""
from scipy import stats
SIGNIFICANCE_LEVEL  = 0.05

H0 = 'As the training fraction increases, the mean accuracies for both DT and RF changes i.e. their performance changes with respect to each other.'
H1 = 'As the training fraction increases, the mean accuracies of DT and RF do not change i.e. their performance remains the same with respect to each other.'

print ('H0: {}'.format(H0))
print ('H1: {}'.format(H1))

# Accuracies from the 10-Fold Cross Validation for each value of t_frac and each model
dt_accs_tfrac_0_05 	= [0.67, 0.69, 0.7, 0.72, 0.72, 0.7, 0.68, 0.73, 0.75, 0.65]
dt_accs_tfrac_0_075 	= [0.65, 0.73, 0.69, 0.68, 0.71, 0.73, 0.68, 0.7, 0.72, 0.71]
dt_accs_tfrac_0_1 	= [0.66, 0.7, 0.7, 0.75, 0.74, 0.72, 0.73, 0.71, 0.72, 0.69]
dt_accs_tfrac_0_15 	= [0.68, 0.73, 0.71, 0.73, 0.7, 0.68, 0.72, 0.7, 0.7, 0.69]
dt_accs_tfrac_0_2 	= [0.68, 0.72, 0.7, 0.75, 0.73, 0.69, 0.7, 0.69, 0.75, 0.72]
dt_accs = [dt_accs_tfrac_0_05, dt_accs_tfrac_0_075, dt_accs_tfrac_0_1, dt_accs_tfrac_0_15, dt_accs_tfrac_0_2]

rf_accs_tfrac_0_05 	= [0.69, 0.74, 0.7, 0.71, 0.73, 0.72, 0.68, 0.73, 0.73, 0.69]
rf_accs_tfrac_0_075 	= [0.68, 0.73, 0.71, 0.75, 0.73, 0.74, 0.7, 0.72, 0.75, 0.7]
rf_accs_tfrac_0_1 	= [0.67, 0.74, 0.71, 0.75, 0.74, 0.74, 0.71, 0.72, 0.72, 0.7]
rf_accs_tfrac_0_15 	= [0.68, 0.77, 0.73, 0.74, 0.75, 0.72, 0.71, 0.73, 0.73, 0.71]
rf_accs_tfrac_0_2 	= [0.69, 0.77, 0.73, 0.75, 0.75, 0.73, 0.73, 0.75, 0.74, 0.71]
rf_accs = [rf_accs_tfrac_0_05, rf_accs_tfrac_0_075, rf_accs_tfrac_0_1, rf_accs_tfrac_0_15, rf_accs_tfrac_0_2]

t_fracs = [0.05, 0.075, 0.1, 0.15, 0.2]
for i in range(len(t_fracs)):
	t_frac = t_fracs[i]
	t_val, p_val = stats.ttest_rel(dt_accs[i], rf_accs[i])
	print ('Fraction: {} H0 for DT and BT: t-statistics = {}, p-value = {} Reject with significance level of {}? {}'.format(t_frac, t_val, p_val, SIGNIFICANCE_LEVEL, (p_val < SIGNIFICANCE_LEVEL)))
