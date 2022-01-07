from pathlib import Path
from tqdm import tqdm
from math import isclose
from console_progressbar import ProgressBar

import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import sys
import json
import concurrent.futures


# pass the --ec2 argument to this script if you're running it on an AWS EC2 instance
IS_RUNNING_ON_EC2 = (len(sys.argv) == 2) and (sys.argv[1] == '--ec2')
DATA_FOLDER = Path('.') if IS_RUNNING_ON_EC2 else Path('..') / 'data' / 'processed' / 'air_quality'


def my_distribution(min_val, max_val, mean, std):
	# EPSILON = sys.float_info.epsilon
	EPSILON = .001
	scale = max_val - min_val
	scale += EPSILON # adding EPSILON to avoid dividing by zero
	location = min_val
	# Mean and standard deviation of the unscaled beta distribution
	unscaled_mean = (mean - min_val) / scale
	unscaled_var = (std / scale) ** 2
	# Computation of alpha and beta can be derived from mean and variance formulas
	t = unscaled_mean / (1 - unscaled_mean)
	# print('[DEBUG]', t, unscaled_var)
	unscaled_var += EPSILON # adding EPSILON to avoid dividing by zero
	beta = ((t / unscaled_var) - (t * t) - (2 * t) - 1) / ((t * t * t) + (3 * t * t) + (3 * t) + 1)
	alpha = beta * t
	# Not all parameters may produce a valid distribution
	if alpha <= 0 or beta <= 0:
		# raise ValueError('Cannot create distribution for the given parameters.')
		# return a uniform distribution as a fallback
		return scipy.stats.uniform(loc=min_val, scale=max_val-min_val) # generates values uniformally in [loc, loc + scale]
	# Make scaled beta distribution with computed parameters
	return scipy.stats.beta(alpha, beta, scale=scale, loc=location)

'''
def dist_to_sample(row):
	# ['date', 'country', 'city', 'specie', 'count', 'min', 'max', 'median', 'variance']
	# according to the data source website:  The data set provides min, max, median and standard deviation for each of the air pollutant species
	# I believe that the last column was mistakenly called variance instead of std
	my_dist = my_distribution(row['min'], row['max'], row['median'], row['variance']) # TODO: fix the last column name: variance -> std
	best_sample_found = None
	smallest_distance = float('inf')
	NB_ITERATIONS = 1
	SAMPLE_SIZE = row['count']
	dist_stats = np.array([row['median'], row['variance']]) # update variance to std here after updating the column name
	for i in range(NB_ITERATIONS):
		sample = my_dist.rvs(size=SAMPLE_SIZE)
		s_min = sample.min()
		s_max = sample.max()
		# s_mean = sample.mean()
		s_median = np.median(sample)
		s_std = sample.std()
		sample_stats = np.array([s_median, s_std])
		distance = np.linalg.norm(dist_stats - sample_stats)
		if distance < smallest_distance:
			smallest_distance = distance
			best_sample_found = sample
	return sample.tolist()
'''

# a faster version, returning the first sample generated
def dist_to_sample(row):
	# ['date', 'country', 'city', 'specie', 'count', 'min', 'max', 'median', 'variance']
	# according to the data source website:  The data set provides min, max, median and standard deviation for each of the air pollutant species
	# I believe that the last column was mistakenly called variance instead of std
	my_dist = my_distribution(row['min'], row['max'], row['median'], row['variance']) # TODO: fix the last column name: variance -> std
	SAMPLE_SIZE = row['count']
	sample = my_dist.rvs(size=SAMPLE_SIZE)
	return sample.tolist()


def save_as_json(json_file_path, data):
	with open(json_file_path, 'w') as f:
		json.dump(data, f)


def read_from_json(json_file_path):
	with open(json_file_path) as f:
		data = json.load(f)
	return data


def process_file(csv_file):
	JSON_FILE_PATH = DATA_FOLDER / f'{csv_file.stem}.json'
	if JSON_FILE_PATH.exists():
		# if the JSON file already there, move on to the next CSV file
		# print(f'{csv_file.stem}.json found, skipping...')
		# continue
		return f'{csv_file.stem}.json found, skipping...'
	# data = []
	save_as_json(JSON_FILE_PATH, []) # create this file and initialize it to an empty list
	print(csv_file.stem)
	df = pd.read_csv(csv_file)
	n_rows = len(df)
	pb = ProgressBar(total=n_rows, prefix='', suffix='', decimals=3, length=50, fill='█', zfill='-')
	# for index, row in tqdm(df.iterrows()):
	for index, row in df.iterrows():
		# print(row)
		# print(dist_to_sample(row))
		pb.print_progress_bar(index + 1)
		sample = dist_to_sample(row)
		# data.append({
		row_data = {
			'date': row['date'],
			'country': row['country'],
			'city': row['city'],
			'specie': row['specie'],
			# 'count': int(row['count']), # make sure to pass an int here!
			# 'min': row['min'],
			# 'max': row['max'],
			# 'median': row['median'],
			# 'std': row['variance'], # TODO: update column name here
			'sample': sample
		}
		# this is to avoid memory limit exceeded errors!
		data = read_from_json(JSON_FILE_PATH)
		data.append(row_data)
		save_as_json(JSON_FILE_PATH, data)
		del data
		# })
		# breakpoint()
		# break
	# break
	# save data as JSON file
	# save_as_json(JSON_FILE_PATH, data)
	# break
	return f'{csv_file.stem}.json saved!'


# for csv_file in DATA_FOLDER.glob('*.csv'):
# 	process_file(csv_file)

with concurrent.futures.ProcessPoolExecutor() as executor:
	results = executor.map(process_file, DATA_FOLDER.glob('*.csv'))
	for result in results:
		print(result)

# print(list(DATA_FOLDER.glob('*.csv')))



# References
# https://stackoverflow.com/questions/35490148/how-to-get-folder-name-in-which-given-file-resides-from-pathlib-path
# https://stackoverflow.com/questions/42513056/how-to-get-absolute-path-of-a-pathlib-path-object
# https://stackoverflow.com/questions/42246819/loop-over-results-from-path-glob-pathlib
# https://stackoverflow.com/questions/9528421/value-for-epsilon-in-python/9528651
# https://math.stackexchange.com/questions/674535/avoid-dividing-by-zero-with-just-variables-and-basic-operators
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html#scipy.stats.uniform
# https://stackoverflow.com/questions/44572109/what-are-the-arguments-for-scipy-stats-uniform
# https://stackoverflow.com/questions/50876840/how-to-get-only-the-name-of-the-path-with-python

# https://www.youtube.com/watch?v=fKl2JW_qrso
# https://www.google.com/search?q=python+add+a+lock+to+a+file
# https://www.google.com/search?q=python+thread+safe+file+write
# https://www.youtube.com/watch?v=Tr4iApndEW0
# https://instances.vantage.sh/?min_memory=8&min_storage=100&selected=z1d.xlarge







