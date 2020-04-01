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
df = pd.read_csv('./wind_station_datasets/wind_station_2/93358-2012.csv', skiprows=3)
wind_data = df['power (MW)']
wind_data = wind_data.values
wind_data = wind_data.reshape(-1, 1)

# Plot the data
# wind_data.plot(subplots=True)
# plt.show()


All_LB = []
Max_iter = 502
Iter_gap = 2

for i in range(Iter_gap, Max_iter, Iter_gap):
    dpmm_one = mixture.BayesianGaussianMixture(n_components=15, max_iter=i, tol=1e-3, n_init=1,
                                               weight_concentration_prior=1/15, warm_start=1, verbose=2).fit(wind_data)
    print('Iteration index: %d \n'
          'Lower bound value: %f ' % (i, dpmm_one.lower_bound_))
    All_LB.append(dpmm_one.lower_bound_)

print('All lower bound values:', All_LB)
scipy.io.savemat('./data/data_lb_502_2_wind2.mat', {'lb_502_2_wind2': All_LB})

# plot
plt.switch_backend('agg')
plt.plot(range(Iter_gap, Max_iter, Iter_gap), All_LB)
plt.savefig('./figures/fig_lb_502_2_wind2.eps', dpi=1000)
# plt.show()

