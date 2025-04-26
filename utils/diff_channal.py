import scipy.io as sio
import os
import matplotlib.pyplot as plt

# Load the data
root_path = './ExtractedFeatures'
feature = sio.loadmat(os.path.join(root_path, '3_20140611.mat'))
key = 'de_movingAve1'
sample = feature[key][31, :, :]

# Plot all columns in one plot with different colors
plt.figure()
num_columns = sample.shape[1]
for i in range(num_columns):
    plt.plot(range(sample.shape[0]), sample[:, i], label=f'Column {i}')
plt.title('Line Plot of All Columns')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.show()

import matplotlib.pyplot as plt

# 获取维度信息
num_time_steps = feature[key].shape[0]
num_columns = feature[key].shape[2]

for i in range(num_columns):
    plt.figure()
    for t in range(num_time_steps):
        sample = feature[key][t, :, :]  # shape: [?, num_columns]
        plt.scatter(
            range(sample.shape[0]),
            sample[:, i],
            s=10,
            color='black'  # 统一颜色，可换成灰色或其他
        )
    plt.title(f'Scatter Plot for Column {i}')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.tight_layout()
    plt.show()
