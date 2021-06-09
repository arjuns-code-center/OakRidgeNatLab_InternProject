import h5py
import numpy as np
from matplotlib import pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

print("Initializing...")
sarsmerscov_train = h5py.File('D:\\ORNL_Code_Data\\sars-mers-cov2_train.h5', 'r')
sarsmerscov_val = h5py.File('D:\\ORNL_Code_Data\\sars-mers-cov2_val.h5', 'r')
label_training = list(open('D:\\ORNL_Coding\\Data Files\\label_train.txt', 'r'))
label_validation = list(open('D:\\ORNL_Coding\\Data Files\\label_val.txt', 'r')) # open all files

print(str(time.ctime()) + ": Successfully loaded all data sets!")

dset_train = np.array(sarsmerscov_train['contact_maps'][:, :, :, 0]).astype(float) # 616207 x 24 x 24
dset_val = np.array(sarsmerscov_val['contact_maps'][:, :, :, 0]).astype(float) # 152052 x 24 x 24
print(dset_train.shape)
print(dset_val.shape)

print(str(time.ctime()) + ": Resized h5 Files!")
print(str(time.ctime()) + ": Plotting...")

plt.figure(1)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(dset_train[i, :, :])

plt.figure(2)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(dset_val[i, :, :])

print(str(time.ctime()) + ": Finished Plotting!")
plt.show()

print(str(time.time()) + ": Initializing Machine Learing...")
# TODO: Write up the ML part on the autoencoder using the 2D data
# train_2D = np.resize(dset_train, (dset_train.shape[0], int((dset_train.shape[1] * dset_train.shape[2]) / 2))) # 616207 x 288
# val_2D = np.resize(dset_val, (dset_val.shape[0], int((dset_val.shape[1] * dset_val.shape[2]) / 2))) # 152052 x 288
print(str(time.time()) + ": Finished Machine Learning!")

