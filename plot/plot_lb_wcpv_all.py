from scipy import io as scio
import matplotlib.pyplot as plt

data_lb_wcpv_wind1 = scio.loadmat('./data/data_lb_wcpv_wind1.mat')
lb_wcpv_wind1A = data_lb_wcpv_wind1['WCP1_LB']
lb_wcpv_wind1B = data_lb_wcpv_wind1['WCP2_LB']
lb_wcpv_wind1A = lb_wcpv_wind1A.reshape(-1, 1)
print(lb_wcpv_wind1A.shape)
lb_wcpv_wind1B = lb_wcpv_wind1B.reshape(-1, 1)
print(lb_wcpv_wind1B.shape)

data_lb_wcpv_wind2 = scio.loadmat('./data/data_lb_wcpv_wind2.mat')
lb_wcpv_wind2A = data_lb_wcpv_wind2['WCP1_LB']
lb_wcpv_wind2B = data_lb_wcpv_wind2['WCP2_LB']
lb_wcpv_wind2A = lb_wcpv_wind2A.reshape(-1, 1)
print(lb_wcpv_wind2A.shape)
lb_wcpv_wind2B = lb_wcpv_wind2B.reshape(-1, 1)
print(lb_wcpv_wind2B.shape)

data_lb_wcpv_wind3 = scio.loadmat('./data/data_lb_wcpv_wind3.mat')
lb_wcpv_wind3A = data_lb_wcpv_wind3['WCP1_LB']
lb_wcpv_wind3B = data_lb_wcpv_wind3['WCP2_LB']
lb_wcpv_wind3A = lb_wcpv_wind3A.reshape(-1, 1)
print(lb_wcpv_wind3A.shape)
lb_wcpv_wind3B = lb_wcpv_wind3B.reshape(-1, 1)
print(lb_wcpv_wind3B.shape)

data_lb_wcpv_wind4 = scio.loadmat('./data/data_lb_wcpv_wind4.mat')
lb_wcpv_wind4A = data_lb_wcpv_wind4['WCP1_LB']
lb_wcpv_wind4B = data_lb_wcpv_wind4['WCP2_LB']
lb_wcpv_wind4A = lb_wcpv_wind4A.reshape(-1, 1)
print(lb_wcpv_wind4A.shape)
lb_wcpv_wind4B = lb_wcpv_wind4B.reshape(-1, 1)
print(lb_wcpv_wind4B.shape)


figure, ax = plt.subplots(figsize=[9, 7])
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 28}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}

iter_max = 302
iter_gap = 2


# plt.bar(range(1, 9), [lb_wcpv_wind1A[-1, -1], lb_wcpv_wind2A[-1, -1],
#                       lb_wcpv_wind3A[-1, -1], lb_wcpv_wind4A[-1, -1], 
#                       lb_wcpv_wind1B[-1, -1], lb_wcpv_wind2B[-1, -1],
#                       lb_wcpv_wind3B[-1, -1], lb_wcpv_wind4B[-1, -1]])

plt.plot(range(iter_gap, iter_max, iter_gap),
         lb_wcpv_wind1A, label='Wind Station 1, WCP=1e-3')
plt.plot(range(iter_gap, iter_max, iter_gap),
         lb_wcpv_wind2A, label='Wind Station 2, WCP=1e-3')
plt.plot(range(iter_gap, iter_max, iter_gap),
         lb_wcpv_wind3A, label='Wind Station 3, WCP=1e-3')
plt.plot(range(iter_gap, iter_max, iter_gap),
         lb_wcpv_wind4A, label='Wind Station 4, WCP=1e-3')
plt.plot(range(iter_gap, iter_max, iter_gap),
         lb_wcpv_wind1B, '--', label='Wind Station 1, WCP=1e3')
plt.plot(range(iter_gap, iter_max, iter_gap),
         lb_wcpv_wind2B, '--', label='Wind Station 2, WCP=1e3')
plt.plot(range(iter_gap, iter_max, iter_gap),
         lb_wcpv_wind3B, '--', label='Wind Station 3, WCP=1e3')
plt.plot(range(iter_gap, iter_max, iter_gap),
         lb_wcpv_wind4B, '--', label='Wind Station 4, WCP=1e3')
plt.legend(prop=font2)
plt.xlabel('Iteration', font1)
plt.ylabel('Lower Bound Value', font1)

plt.tick_params(labelsize=20)
# 设置坐标刻度值的字体
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

# plt.title('', fontsize=12)
plt.show()
