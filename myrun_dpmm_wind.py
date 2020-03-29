from __future__ import division
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd

from sklearn import mixture


# Guide of pandas
# df = pd.read_csv('./wind_power_datasets/windfarm1.csv', skiprows=3)
# df.head()
# df.tail()
# uni_data = df['power (MW)']
# # uni_data.index = df['Date Time']
# uni_data.head()
# uni_data.tail()
# uni_data.plot(subplots=True)
# uni_data = uni_data.values

df = pd.read_csv('./wind_station_datasets/wind_station_2/93358-2008.csv', skiprows=3)
# print(df.head())
# print(df.tail())

wind_data = df['power (MW)']
print(wind_data.head())


# wind_data.plot(subplots=True)
# plt.show()


wind_data = wind_data.values
wind_data = wind_data.reshape(-1, 1)



# Wind farm 1
# Fit the normed wind speed data by GMM via EM
# gmm_1_one = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(y_one_used)
# sample_gmm_1_one, label_gmm_1_one = gmm_1_one.sample(n_samples=10000)
# gmm_2_one = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(y_one_used)
# sample_gmm_2_one, label_gmm_2_one = gmm_2_one.sample(n_samples=10000)
# gmm_3_one = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(y_one_used)
# sample_gmm_3_one, label_gmm_3_one = gmm_3_one.sample(n_samples=10000)
# gmm_4_one = mixture.GaussianMixture(n_components=4, covariance_type='full').fit(y_one_used)
# sample_gmm_4_one, label_gmm_4_one = gmm_4_one.sample(n_samples=10000)
# gmm_5_one = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(y_one_used)
# sample_gmm_5_one, label_gmm_5_one = gmm_5_one.sample(n_samples=10000)

All_LB = []
Max_iter = 22
Iter_gap = 2

for i in range(Iter_gap, Max_iter, Iter_gap):
    dpmm_one = mixture.BayesianGaussianMixture(n_components=15, max_iter=i, tol=1e-3, n_init=1,
                                               weight_concentration_prior=1/15, warm_start=1, verbose=2).fit(wind_data)
    print('Iteration index: %d \n'
          'Lower bound value: %f ' % (i, dpmm_one.lower_bound_))
    All_LB.append(dpmm_one.lower_bound_)

print('All lower bound values:', All_LB)

plt.plot(range(Iter_gap, Max_iter, Iter_gap), All_LB)
plt.show()



# sample_dpmm_one, label_dpmm_one = dpmm_one.sample(n_samples=10000)

# # Save data
# scipy.io.savemat('gmm_1_one.mat', {'n_components': gmm_1_one.n_components,'weights': gmm_1_one.weights_, 'means': gmm_1_one.means_, 'covariances': gmm_1_one.covariances_, 'sample_gmm_1_one':sample_gmm_1_one})
#
# scipy.io.savemat('gmm_2_one.mat', {'n_components': gmm_2_one.n_components,'weights': gmm_2_one.weights_, 'means': gmm_2_one.means_, 'covariances': gmm_2_one.covariances_, 'sample_gmm_2_one':sample_gmm_2_one})
#
# scipy.io.savemat('gmm_3_one.mat', {'n_components': gmm_3_one.n_components,'weights': gmm_3_one.weights_, 'means': gmm_3_one.means_, 'covariances': gmm_3_one.covariances_, 'sample_gmm_3_one':sample_gmm_3_one})
#
# scipy.io.savemat('gmm_4_one.mat', {'n_components': gmm_4_one.n_components,'weights': gmm_4_one.weights_, 'means': gmm_4_one.means_, 'covariances': gmm_4_one.covariances_, 'sample_gmm_4_one':sample_gmm_4_one})
#
# scipy.io.savemat('gmm_5_one.mat', {'n_components': gmm_5_one.n_components,'weights': gmm_5_one.weights_, 'means': gmm_5_one.means_, 'covariances': gmm_5_one.covariances_, 'sample_gmm_5_one':sample_gmm_5_one})
#
# scipy.io.savemat('dpmm_one.mat', {'n_components': dpmm_one.n_components,'weights': dpmm_one.weights_, 'means': dpmm_one.means_, 'covariances': dpmm_one.covariances_, 'sample_dpmm_one':sample_dpmm_one})

print("Test1")