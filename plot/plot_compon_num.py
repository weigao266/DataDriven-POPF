
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



for i in range(1,16):
    gmm_wind_1_diffnum = mixture.BayesianGaussianMixture(n_components=i,
                                        covariance_type='full', weight_concentration_prior_type='dirichlet_process', tol=1e-3, max_iter=100).fit(y_one_used)

    scipy.io.savemat('gmm_wind_1_com_%d.mat'%(i), {'n_components': gmm_wind_1_diffnum.n_components,'weights': gmm_wind_1_diffnum.weights_, 'means': gmm_wind_1_diffnum.means_, 'covariances': gmm_wind_1_diffnum.covariances_})



for i in range(1,16):
    dpmm_wind_1_diffnum = mixture.GaussianMixture(n_components=i, covariance_type='full', tol=1e-3, max_iter=100).fit(y_one_used)

    scipy.io.savemat('dpmm_wind_1_com_%d.mat'%(i), {'n_components': dpmm_wind_1_diffnum.n_components,'weights': dpmm_wind_1_diffnum.weights_, 'means': dpmm_wind_1_diffnum.means_, 'covariances': dpmm_wind_1_diffnum.covariances_})



