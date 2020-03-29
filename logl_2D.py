
from __future__ import division
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import scipy.io

from sklearn import mixture
import time

# Wind
y_norm_one_used = scipy.io.loadmat('y_norm_one_used.mat')['y_norm_one_used']
y_norm_two_used = scipy.io.loadmat('y_norm_two_used.mat')['y_norm_two_used']


# Consider the correlation of wind farm 1 2
norm_data_wind_12 = np.column_stack((y_norm_one_used, y_norm_two_used))

# norm_gmm

start = time.time()
norm_gmm_1_wind_12 = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(norm_data_wind_12)
end = time.time()
print "time_norm_gmm_1_wind_12 =", end-start

start = time.time()
norm_gmm_2_wind_12 = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(norm_data_wind_12)
end = time.time()
print "time_norm_gmm_2_wind_12 =", end-start

start = time.time()
norm_gmm_3_wind_12 = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(norm_data_wind_12)
end = time.time()
print "time_norm_gmm_3_wind_12 =", end-start

start = time.time()
norm_gmm_4_wind_12 = mixture.GaussianMixture(n_components=4, covariance_type='full').fit(norm_data_wind_12)
end = time.time()
print "time_norm_gmm_4_wind_12 =", end-start

scipy.io.savemat('norm_gmm_1_wind_12.mat', {'n_components': norm_gmm_1_wind_12.n_components,'weights': norm_gmm_1_wind_12.weights_, 'means': norm_gmm_1_wind_12.means_, 'covariances': norm_gmm_1_wind_12.covariances_})

scipy.io.savemat('norm_gmm_2_wind_12.mat', {'n_components': norm_gmm_2_wind_12.n_components,'weights': norm_gmm_2_wind_12.weights_, 'means': norm_gmm_2_wind_12.means_, 'covariances': norm_gmm_2_wind_12.covariances_})

scipy.io.savemat('norm_gmm_3_wind_12.mat', {'n_components': norm_gmm_3_wind_12.n_components,'weights': norm_gmm_3_wind_12.weights_, 'means': norm_gmm_3_wind_12.means_, 'covariances': norm_gmm_3_wind_12.covariances_})

scipy.io.savemat('norm_gmm_4_wind_12.mat', {'n_components': norm_gmm_4_wind_12.n_components,'weights': norm_gmm_4_wind_12.weights_, 'means': norm_gmm_4_wind_12.means_, 'covariances': norm_gmm_4_wind_12.covariances_})


# norm_dpmm
start = time.time()
norm_dpmm_wind_12 = mixture.BayesianGaussianMixture(n_components=15,
                                        covariance_type='full', weight_concentration_prior_type='dirichlet_process').fit(norm_data_wind_12)
end = time.time()
print "time_norm_dpmm_wind_12 =", end-start

scipy.io.savemat('norm_dpmm_wind_12.mat', {'n_components': norm_dpmm_wind_12.n_components,'weights': norm_dpmm_wind_12.weights_, 'means': norm_dpmm_wind_12.means_, 'covariances': norm_dpmm_wind_12.covariances_})
