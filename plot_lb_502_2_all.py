from scipy import io as scio
import matplotlib.pyplot as plt

data_lb_502_2_wind1 = scio.loadmat('./data/data_lb_502_2_wind1.mat')
lb_502_2_wind1 = data_lb_502_2_wind1['lb_502_2_wind1']
lb_502_2_wind1 = lb_502_2_wind1.reshape(-1, 1)
print(lb_502_2_wind1.shape)

data_lb_502_2_wind2 = scio.loadmat('./data/data_lb_502_2_wind2.mat')
lb_502_2_wind2 = data_lb_502_2_wind2['lb_502_2_wind2']
lb_502_2_wind2 = lb_502_2_wind2.reshape(-1, 1)
print(lb_502_2_wind2.shape)

data_lb_502_2_wind3 = scio.loadmat('./data/data_lb_502_2_wind3.mat')
lb_502_2_wind3 = data_lb_502_2_wind3['lb_502_2_wind3']
lb_502_2_wind3 = lb_502_2_wind3.reshape(-1, 1)
print(lb_502_2_wind3.shape)

data_lb_502_2_wind4 = scio.loadmat('./data/data_lb_502_2_wind4.mat')
lb_502_2_wind4 = data_lb_502_2_wind4['lb_502_2_wind4']
lb_502_2_wind4 = lb_502_2_wind4.reshape(-1, 1)
print(lb_502_2_wind4.shape)


figure, ax = plt.subplots(figsize=[9, 7])
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 28}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 24}

plt.plot(range(2, 502, 2), lb_502_2_wind1, label='Wind Station 1')
plt.plot(range(2, 502, 2), lb_502_2_wind2, label='Wind Station 2')
plt.plot(range(2, 502, 2), lb_502_2_wind3, label='Wind Station 3')
plt.plot(range(2, 502, 2), lb_502_2_wind4, label='Wind Station 4')
plt.legend(prop=font2)
plt.xlabel('Iteration', font1)
plt.ylabel('Lower Bound Value', font1)

plt.tick_params(labelsize=20)
# 设置坐标刻度值的字体
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

# plt.title('', fontsize=12)
plt.show()
