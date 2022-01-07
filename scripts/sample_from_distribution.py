
# https://stackoverflow.com/questions/50626710/generating-random-numbers-with-predefined-mean-std-min-and-max
# https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from math import isclose

def my_distribution(min_val, max_val, mean, std):
    scale = max_val - min_val
    location = min_val
    # Mean and standard deviation of the unscaled beta distribution
    unscaled_mean = (mean - min_val) / scale
    unscaled_var = (std / scale) ** 2
    # Computation of alpha and beta can be derived from mean and variance formulas
    t = unscaled_mean / (1 - unscaled_mean)
    beta = ((t / unscaled_var) - (t * t) - (2 * t) - 1) / ((t * t * t) + (3 * t * t) + (3 * t) + 1)
    alpha = beta * t
    # Not all parameters may produce a valid distribution
    if alpha <= 0 or beta <= 0:
        raise ValueError('Cannot create distribution for the given parameters.')
    # Make scaled beta distribution with computed parameters
    return scipy.stats.beta(alpha, beta, scale=scale, loc=location)

np.random.seed(100)

min_val = 1.5
max_val = 35
mean = 9.87
std = 3.1

print('distribution parameters:')
print('mean:', mean, 'std:', std)
print('min:', min_val, 'max:', max_val)

my_dist = my_distribution(min_val, max_val, mean, std)

# Plot distribution PDF
# x = np.linspace(min_val, max_val, 100)
# f = plt.plot(x, my_dist.pdf(x)) # Probability Density Function
# plt.show()
# print(f) # to make sure that plt.plot(.) return a figure object
# import sys
# sys.exit()

# Stats
print('\nmean:', my_dist.mean(), 'std:', my_dist.std())

# Get a large sample to check bounds
# sample = my_dist.rvs(size=100000) # generate a sequence of random variates
# print('min:', sample.min(), 'max:', sample.max())

# Idea: sample until the sample has the same statistical measures of the distribution we're sampling from
'''
samples = np.array([])
PRECISION = 2 # precision of n significant digits
ERROR_TOLERANCE = 10**-PRECISION
MAX_NB_ITERATIONS = 9*10**3
n_iterations = 0
while True:
    sample = my_dist.rvs(size=1) # sample a new value
    samples = np.concatenate((samples, sample)) # add it to the samples list
    s_min = samples.min()
    s_max = samples.max()
    s_mean = samples.mean()
    s_std = samples.std()
    print('s_min', s_min, 's_max', s_max, 's_mean', s_mean, 's_std', s_std)
    if all([
        # let's ignore s_min and s_max and check for s_mean and s_std only instead
        # why? requiring to get both s_min and s_max will slow down the algorithm
        # and it's less likely to get those values every time
        # isclose(s_min,   min_val,  abs_tol=ERROR_TOLERANCE),
        # isclose(s_max,   max_val,  abs_tol=ERROR_TOLERANCE),
        isclose(s_mean,  mean,     abs_tol=ERROR_TOLERANCE),
        isclose(s_std,   std,      abs_tol=ERROR_TOLERANCE),
    ]):
        break
    n_iterations += 1
    # to avoid time-consuming loop, we set a hard stop of at least 3K iterations
    # why that value in particular? because we noticed that the 'count'
    # column's values didn't exceed 3000
    if n_iterations == MAX_NB_ITERATIONS:
        print('maximum number of iterations reached!')
        break
print(samples.size, 'values sampled')

print('\ndistribution parameters:')
print('min', min_val, 'max', max_val, 'mean', mean, 'std', std)
'''

# Another idea to try, for a constant number of iterations sample a constant number of numbers
# then compute the distance between the descriptive statistics of the sample and the generating
# distribution then pick the closest one to the distribution (the smallest distance)

best_sample_found = None
smallest_distance = float('inf')

NB_ITERATIONS = 10**3
SAMPLE_SIZE = 3000
# dist_stats = np.array([min_val, max_val, mean, std])
dist_stats = np.array([mean, std])
print('dist_stats', dist_stats)
for i in range(NB_ITERATIONS):
    sample = my_dist.rvs(size=SAMPLE_SIZE)
    s_min = sample.min()
    s_max = sample.max()
    s_mean = sample.mean()
    s_std = sample.std()
    # sample_stats = np.array([s_min, s_max, s_mean, s_std])
    sample_stats = np.array([s_mean, s_std])
    distance = np.linalg.norm(dist_stats - sample_stats)
    if distance < smallest_distance:
        smallest_distance = distance
        best_sample_found = sample
        print('new best sample found!', sample_stats, distance)

print('\nbest sample:', sample)

# References
# https://stackoverflow.com/questions/558216/function-to-determine-if-two-numbers-are-nearly-equal-when-rounded-to-n-signific








