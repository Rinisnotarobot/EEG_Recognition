import scipy.io as sio
import os
import matplotlib.pyplot as plt

# Load the data
root_path = '../ExtractedFeatures'
feature = sio.loadmat(os.path.join(root_path, '1_20131027.mat'))
key = 'de_movingAve1'
sample = feature[key][0, :, :]

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