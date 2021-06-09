import h5py
import numpy as np
from matplotlib import pyplot as plt
import sklearn

sarsmerscov_train = h5py.File('D:\\ORNL_Code_Data\\sars-mers-cov2_train.h5', 'r')
sarsmerscov_val = h5py.File('D:\\ORNL_Code_Data\\sars-mers-cov2_val.h5', 'r')
label_training = list(open('label_train.txt', 'r'))
label_validation = list(open('label_val.txt', 'r')) # open all files

print("Successfully loaded all data sets!")

dset_train = np.array(sarsmerscov_train['contact_maps']) # 616207 x 24 x 24 x 1
dset_val = np.array(sarsmerscov_val['contact_maps']) # 152052 x 24 x 24 x 1
dset_train.resize((dset_train.shape[0], int((dset_train.shape[1] * dset_train.shape[2]) / 2))) # 616207 x 288
dset_val.resize((dset_val.shape[0], int((dset_val.shape[1] * dset_val.shape[2]) / 2))) # 152052 x 288

print("Resized h5 Files")

sars_train = np.zeros((dset_train.shape[0])).astype(int)
mers_train = np.zeros((dset_train.shape[0])).astype(int)
cov_train = np.zeros((dset_train.shape[0])).astype(int)
for i in range(dset_train.shape[0]):
    num = int(str(label_training[i]).strip('\n'))
    if num == 0: cov_train[i] = int(sum(dset_train[i, :]))
    elif num == 1: mers_train[i] = int(sum(dset_train[i, :]))
    elif num == 2: sars_train[i] = int(sum(dset_train[i, :])) # find total number of contacts in each sample

print("Finished loading training sets")

sars_val = np.zeros((dset_val.shape[0])).astype(int)
mers_val = np.zeros((dset_val.shape[0])).astype(int)
cov_val = np.zeros((dset_val.shape[0])).astype(int)
for i in range(dset_val.shape[0]):
    num = int(str(label_validation[i]).strip('\n'))
    if num == 0: cov_val[i] = int(sum(dset_val[i, :]))
    elif num == 1: mers_val[i] = int(sum(dset_val[i, :]))
    elif num == 2: sars_val[i] = int(sum(dset_val[i, :])) # find total number of contacts in each sample

print("Finished loading validation sets")
print("Plotting...")

train_fig = plt.figure(1)
train_fig.add_subplot(111)
plt.scatter(np.arange(0, dset_train.shape[0]), sars_train, c='b')
plt.scatter(np.arange(0, dset_train.shape[0]), mers_train, c='r')
plt.scatter(np.arange(0, dset_train.shape[0]), cov_train, c='k')
plt.ylabel('Total Contacts')
plt.xlabel('Samples (Training Set)')

val_fig = plt.figure(2)
val_fig.add_subplot(111)
plt.scatter(np.arange(0, dset_val.shape[0]), sars_val, c='b')
plt.scatter(np.arange(0, dset_val.shape[0]), mers_val, c='r')
plt.scatter(np.arange(0, dset_val.shape[0]), cov_val, c='k')
plt.ylabel('Total Contacts')
plt.xlabel('Samples (Validation Set)')

print("Finished plotting")
plt.show() # plot both sets