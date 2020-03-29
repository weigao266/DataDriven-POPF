
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


# norm_gmm

norm_gmm_1_wind = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(y_norm_one_used)

norm_gmm_2_wind = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(y_norm_one_used)

scipy.io.savemat('norm_gmm_1_wind.mat', {'n_components': norm_gmm_1_wind.n_components,'weights': norm_gmm_1_wind.weights_, 'means': norm_gmm_1_wind.means_, 'covariances': norm_gmm_1_wind.covariances_})

scipy.io.savemat('norm_gmm_2_wind.mat', {'n_components': norm_gmm_2_wind.n_components,'weights': norm_gmm_2_wind.weights_, 'means': norm_gmm_2_wind.means_, 'covariances': norm_gmm_2_wind.covariances_})
