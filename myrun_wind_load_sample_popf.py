from __future__ import division
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import scipy.io

from sklearn import mixture
import time


# normed wind power data
y_one_used = scipy.io.loadmat('y_one_used.mat')['y_one_used']
y_two_used = scipy.io.loadmat('y_two_used.mat')['y_two_used']
y_three_used = scipy.io.loadmat('y_three_used.mat')['y_three_used']
y_four_used = scipy.io.loadmat('y_four_used.mat')['y_four_used']
y_five_used = scipy.io.loadmat('y_five_used.mat')['y_five_used']
y_six_used = scipy.io.loadmat('y_six_used.mat')['y_six_used']

# normed load power data
x_one_zone = scipy.io.loadmat('x_one_zone.mat')['x_one_zone']
x_two_zone = scipy.io.loadmat('x_two_zone.mat')['x_two_zone']
x_three_zone = scipy.io.loadmat('x_three_zone.mat')['x_three_zone']
x_four_zone = scipy.io.loadmat('x_four_zone.mat')['x_four_zone']


# Case118
# Consider the correlation of wind farm 1 2 3 4 5 6
data_wind_123456 = np.column_stack((y_one_used, y_two_used, y_three_used, y_four_used, y_five_used, y_six_used))

# Consider the correlation of load zone 1 2 3 4
data_load_1234 = np.column_stack((x_one_zone[0:1440], x_two_zone, x_three_zone, x_four_zone))

# wind farm 1 2 3 4 5 6
# DPMM via Variational Inference
start = time.time()
dpmm_wind_123456 = mixture.BayesianGaussianMixture(n_components=15,
                                        covariance_type='full', weight_concentration_prior_type='dirichlet_process').fit(data_wind_123456)
sample_wind_power_dpmm_123456, label_wind_power_dpmm_123456 = dpmm_wind_123456.sample(n_samples=10000)
end = time.time()
print "time_dpmm_VI_wind_123456 =", end-start



## Load zone 1 2 3 4
# DPMM via Variational Inference
dpmm_load_1234 = mixture.BayesianGaussianMixture(n_components=15,
                                        covariance_type='full', weight_concentration_prior_type='dirichlet_process').fit(data_load_1234)
sample_load_dpmm_1234, label_load_dpmm_1234 = dpmm_load_1234.sample(n_samples=10000)



# Save data
scipy.io.savemat('popf_118_wind_power_sample.mat', {'sample_wind_power_dpmm_123456': sample_wind_power_dpmm_123456})

scipy.io.savemat('popf_118_load_sample.mat', {'sample_load_dpmm_1234': sample_load_dpmm_1234})

plt.show()