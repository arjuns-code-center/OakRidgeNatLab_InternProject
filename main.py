import h5py
import numpy as np
from matplotlib import pyplot as plt
import sklearn

sarsmerscov_train = h5py.File('D:\\ORNL_Code_Data\\sars-mers-cov2_train.h5', 'r')
sarsmerscov_val = h5py.File('D:\\ORNL_Code_Data\\sars-mers-cov2_val.h5', 'r')
label_training = list(open('label_train.txt', 'r'))
label_validation = list(open('label_val.txt', 'r')) # open all files

print(sarsmerscov_train.keys()) # print all the keys in dataset
print(sarsmerscov_val.keys())

dset_train = np.array(sarsmerscov_train['contact_maps'])
dset_val = np.array(sarsmerscov_val['contact_maps'])
print(dset_train.shape)
print(dset_val.shape) # print the shape of each dataset in samples, feature, feature, 1

dset_train.resize((dset_train.shape[0], int((dset_train.shape[1] * dset_train.shape[2]) / 2)))
dset_val.resize((dset_val.shape[0], int((dset_val.shape[1] * dset_val.shape[2]) / 2)))

print(dset_train.shape)
print(dset_val.shape)

sars = []
mers = []
cov = []
for i in range(len(dset_train)):
    num = int(str(label_training[i]).strip('\n'))
    if num == 0: cov.append(dset_train[i, :])
    elif num == 1: mers.append(dset_train[i, :])
    elif num == 2: sars.append(dset_train[i, :])

print(len(cov))
print(len(mers))
print(len(sars)) # print out the lengths

rowsCov, colsCov = np.where(np.array(cov).astype(int) == 1)
rowsMers, colsMers = np.where(np.array(mers).astype(int) == 1)
rowsSars, colsSars = np.where(np.array(sars).astype(int) == 1)

plt.scatter(rowsCov, colsCov, 'b')
plt.scatter(rowsMers, colsMers, 'r')
plt.scatter(rowsSars, colsSars, 'k')
plt.xlabel('Contacts')
plt.ylabel('Samples')
plt.show()
