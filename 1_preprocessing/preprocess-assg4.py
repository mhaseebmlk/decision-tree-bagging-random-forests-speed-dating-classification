import sys

import pandas as pd
import numpy as np

DATA_FILE_NAME = 'dating-full.csv'
DATA_PATH = '../dataset/' + DATA_FILE_NAME
OUTPUT_FILE_NAME_TRAINING = 'trainingSet.csv'
OUTPUT_FILE_NAME_TEST = 'testSet.csv'

RANDOM_STATE = 47
FRAC = 0.2

continuous_valued_columns = ['age', 'age_o', 'importance_same_race', 'importance_same_religion', 'pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests', 'attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important', 'attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'attractive_partner', 'sincere_partner', 'intelligence_parter', 'funny_partner', 'ambition_partner', 'shared_interests_partner', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'interests_correlate', 'expected_happy_with_sd_people', 'like']

# continuous valued column ranges as mentioend in the field-meaning.pdf. both values are inclusive
col_ranges = {'age': (18, 58), 'age_o': (18, 58), 'importance_same_race': (0, 10), 'importance_same_religion': (0, 10), 'pref_o_attractive': (0, 1), 'pref_o_sincere': (0, 1), 'pref_o_intelligence': (0, 1), 'pref_o_funny': (0, 1), 'pref_o_ambitious': (0, 1), 'pref_o_shared_interests': (0, 1), 'attractive_important': (0, 1), 'sincere_important': (0, 1), 'intelligence_important': (0, 1), 'funny_important': (0, 1), 'ambition_important': (0, 1), 'shared_interests_important': (0, 1), 'attractive': (0, 10), 'sincere': (0, 10), 'intelligence': (0, 10), 'funny': (0, 10), 'ambition': (0, 10), 'attractive_partner': (0, 10), 'sincere_partner': (0, 10), 'intelligence_parter': (0, 10), 'funny_partner': (0, 10), 'ambition_partner': (0, 10), 'shared_interests_partner': (0, 10), 'sports': (0, 10), 'tvsports': (0, 10), 'exercise': (0, 10), 'dining': (0, 10), 'museums': (0, 10), 'art': (0, 10), 'hiking': (0, 10), 'gaming': (0, 10), 'clubbing': (0, 10), 'reading': (0, 10), 'tv': (0, 10), 'theater': (0, 10), 'movies': (0, 10), 'concerts': (0, 10), 'music': (0, 10), 'shopping': (0, 10), 'yoga': (0, 10), 'interests_correlate': (-1, 1), 'expected_happy_with_sd_people': (0, 10), 'like': (0, 10)}

def perform_label_encoding(column):
    column = column.astype('category')
    codes_for_column = {}
    for i, category in enumerate(column.cat.categories):
        codes_for_column[category] = i
    return column.cat.codes

def update_anomalous_max_values(row, col_min, col_max , true_range):
	true_max = true_range[1]
	row = true_max if row > true_max else row
	return row

def bin_row(row, intervals):
	bin_num = intervals.index(row)
	return bin_num

def discretize(df, num_bins, output_file_name=None, print_=True):
	bin_interval_mappings = dict()
	for col in continuous_valued_columns:
		col_min, col_max = df[col].min(), df[col].max()
		true_range = col_ranges[col]
		df[col] = df[col].apply(update_anomalous_max_values, args=(col_min,col_max, true_range ))

		true_min, true_max = true_range[0], true_range[1]
		bin_min_val = -0.01 if true_min==0 else (true_min-(0.1/100 * true_min))

		bins = list(np.linspace(true_min, true_max, num_bins+1))
		bins = list(zip(bins, bins[1:]))
		bins[0] = (bin_min_val,bins[0][1])
		bins = pd.IntervalIndex.from_tuples(bins)

		bin_interval_mappings[col] = {bin: interval for bin, interval in enumerate(bins)}

		bins_list = bins.tolist()
		df[col] = pd.cut(df[col], bins).apply(bin_row, intervals= bins_list)

		counts = [len(df[df[col] == i])  for i in range(len(bins_list))]

		if print_==True:
			print ('{}: {}'.format(col, counts))

	if output_file_name != None:
		df.to_csv(output_file_name, index=False)

	return df

# Read the first 6500 rows from the dataset
df = pd.read_csv(DATA_PATH, nrows=6500)
decision_col = df['decision']

# Drop the columns
columns_to_drop = ['race', 'race_o', 'field']
df = df.drop(columns=columns_to_drop)

#Label encode
df[['gender']] = df[['gender']].apply(perform_label_encoding)

# Binning
num_bins = 2
df = discretize(df, num_bins, output_file_name=None, print_=False)

#Move the target class to the end
df = df.drop(['decision'], axis = 1)
df['decision'] = decision_col

# Sampling Test and Training sets
df_test = df.sample(frac=FRAC, random_state=RANDOM_STATE)
df_train = df.drop(df_test.index)

df_test.to_csv(OUTPUT_FILE_NAME_TEST, index=False)
df_train.to_csv(OUTPUT_FILE_NAME_TRAINING, index=False)
