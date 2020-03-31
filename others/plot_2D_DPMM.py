
from __future__ import division
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import scipy.io

from sklearn import mixture
import time

# Wind
y_one_used = scipy.io.loadmat('y_one_used.mat')['y_one_used']
y_two_used = scipy.io.loadmat('y_two_used.mat')['y_two_used']
y_three_used = scipy.io.loadmat('y_three_used.mat')['y_three_used']
y_four_used = scipy.io.loadmat('y_four_used.mat')['y_four_used']
y_five_used = scipy.io.loadmat('y_five_used.mat')['y_five_used']
y_six_used = scipy.io.loadmat('y_six_used.mat')['y_six_used']


# Consider the correlation of wind farm 1 2 and 3 4 and 5 6
data_wind_12 = np.column_stack((y_one_used, y_two_used))
data_wind_34 = np.column_stack((y_three_used, y_four_used))
data_wind_56 = np.column_stack((y_five_used, y_six_used))

dpmm_wind_12 = mixture.BayesianGaussianMixture(n_components=15,
                                        covariance_type='full', weight_concentration_prior_type='dirichlet_process').fit(data_wind_12)

scipy.io.savemat('dpmm_wind_12.mat', {'n_components': dpmm_wind_12.n_components,'weights': dpmm_wind_12.weights_, 'means': dpmm_wind_12.means_, 'covariances': dpmm_wind_12.covariances_})




# Load
x_one_zone = scipy.io.loadmat('x_one_zone.mat')['x_one_zone']
x_two_zone = scipy.io.loadmat('x_two_zone.mat')['x_two_zone']
x_three_zone = scipy.io.loadmat('x_three_zone.mat')['x_three_zone']
x_four_zone = scipy.io.loadmat('x_four_zone.mat')['x_four_zone']


# Consider the correlation of Load 3 4
data_load_12 = np.column_stack((x_one_zone[0:1440], x_two_zone))


dpmm_load_12 = mixture.BayesianGaussianMixture(n_components=15,
                                        covariance_type='full', weight_concentration_prior_type='dirichlet_process').fit(data_load_12)

scipy.io.savemat('dpmm_load_12.mat', {'n_components': dpmm_load_12.n_components,'weights': dpmm_load_12.weights_, 'means': dpmm_load_12.means_, 'covariances': dpmm_load_12.covariances_})