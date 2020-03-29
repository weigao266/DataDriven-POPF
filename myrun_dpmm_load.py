from __future__ import division
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import scipy.io

from sklearn import mixture


# load normed wind speed data
x_two_zone = scipy.io.loadmat('x_two_zone.mat')['x_two_zone']


# Do not consider the correlation

# Wind farm 1
# Fit the normed wind speed data by GMM via EM
gmm_1_two = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(x_two_zone)
sample_gmm_1_two, label_gmm_1_two = gmm_1_two.sample(n_samples=10000)
gmm_2_two = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(x_two_zone)
sample_gmm_2_two, label_gmm_2_two = gmm_2_two.sample(n_samples=10000)
gmm_3_two = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(x_two_zone)
sample_gmm_3_two, label_gmm_3_two = gmm_3_two.sample(n_samples=10000)
gmm_4_two = mixture.GaussianMixture(n_components=4, covariance_type='full').fit(x_two_zone)
sample_gmm_4_two, label_gmm_4_two = gmm_4_two.sample(n_samples=10000)
gmm_5_two = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(x_two_zone)
sample_gmm_5_two, label_gmm_5_two = gmm_5_two.sample(n_samples=10000)


# Fit the normed wind speed data by DPMM via Variational Inference
dpmm_two = mixture.BayesianGaussianMixture(n_components=15,
                                        covariance_type='full', weight_concentration_prior=1/50).fit(x_two_zone)
sample_dpmm_two, label_dpmm_two = dpmm_two.sample(n_samples=10000)


# Save data
scipy.io.savemat('gmm_load_1_two.mat', {'n_components': gmm_1_two.n_components,'weights': gmm_1_two.weights_, 'means': gmm_1_two.means_, 'covariances': gmm_1_two.covariances_, 'sample_gmm_1_two':sample_gmm_1_two})

scipy.io.savemat('gmm_load_2_two.mat', {'n_components': gmm_2_two.n_components,'weights': gmm_2_two.weights_, 'means': gmm_2_two.means_, 'covariances': gmm_2_two.covariances_, 'sample_gmm_2_two':sample_gmm_2_two})

scipy.io.savemat('gmm_load_3_two.mat', {'n_components': gmm_3_two.n_components,'weights': gmm_3_two.weights_, 'means': gmm_3_two.means_, 'covariances': gmm_3_two.covariances_, 'sample_gmm_3_two':sample_gmm_3_two})

scipy.io.savemat('gmm_load_4_two.mat', {'n_components': gmm_4_two.n_components,'weights': gmm_4_two.weights_, 'means': gmm_4_two.means_, 'covariances': gmm_4_two.covariances_, 'sample_gmm_4_two':sample_gmm_4_two})

scipy.io.savemat('gmm_load_5_two.mat', {'n_components': gmm_5_two.n_components,'weights': gmm_5_two.weights_, 'means': gmm_5_two.means_, 'covariances': gmm_5_two.covariances_, 'sample_gmm_5_two':sample_gmm_5_two})

scipy.io.savemat('dpmm_load_two.mat', {'n_components': dpmm_two.n_components,'weights': dpmm_two.weights_, 'means': dpmm_two.means_, 'covariances': dpmm_two.covariances_, 'sample_dpmm_two':sample_dpmm_two})

plt.show()

print("T2")