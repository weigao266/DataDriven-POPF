from __future__ import division
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd

from sklearn import mixture

# wind_station_1: 95485
# wind_station_2: 93358
# wind_station_3: 98359
# wind_station_4: 100373
df = pd.read_csv('./wind_station_datasets/wind_station_4/100373-2012.csv', skiprows=3)
wind_data = df['power (MW)']
wind_data = wind_data.values
wind_data = wind_data.reshape(-1, 1)

# Plot the data
# wind_data.plot(subplots=True)
# plt.show()

A_LB = []
B_LB = []
iter_max = 302
iter_gap = 2
nc = 15
wcp_list = [1e-3, 1e3]
for wcp in wcp_list:
    LBs = []
    for i in range(iter_gap, iter_max, iter_gap):
        dpmm_one = mixture.BayesianGaussianMixture(n_components=nc, max_iter=i, tol=1e-3, n_init=1,
                                               weight_concentration_prior=wcp, warm_start=1, verbose=2).fit(wind_data)

        LBs.append(dpmm_one.lower_bound_)
    if wcp == 1e-3:
        A_LB = LBs
    else:
        B_LB = LBs

print(A_LB, '\n', B_LB)

# save data
scipy.io.savemat('./data/data_lb_wcpv_wind4.mat', {'WCP1_LB': A_LB, 'WCP2_LB': B_LB})

# plot
plt.switch_backend('agg')
line1 = plt.plot(range(iter_gap, iter_max, iter_gap), A_LB, label='WCP=1e-3')
line2 = plt.plot(range(iter_gap, iter_max, iter_gap), B_LB, label='WCP=1e3')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Lower Bound Value')
plt.savefig('./figures/fig_lb_wcpv_wind4.eps', dpi=1000)
# plt.show()




#
# plt.switch_backend('agg')
# plt.plot(range(Iter_gap, Max_iter, Iter_gap), All_LB)
#
# # plt.show()

