# Summer 2021 Internship project with Oak Ride National Laboratory (ORNL)
# Code Written By: Arjun Viswanathan
# Mentored By: Dr. Junqi Yin
# Date Started: 6/7/2021
# Date TBC: 8/13/2021
# All datasets provided by Dr. Yin

import h5py
import numpy as np
from matplotlib import pyplot as plt
import time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

def dataLoading():
    sarsmerscov_train = h5py.File('D:\\ORNL_Code_Data\\sars-mers-cov2_train.h5', 'r')
    sarsmerscov_val = h5py.File('D:\\ORNL_Code_Data\\sars-mers-cov2_val.h5', 'r')
    label_training = list(open('D:\\ORNL_Coding\\Data Files\\label_train.txt', 'r'))
    label_validation = list(open('D:\\ORNL_Coding\\Data Files\\label_val.txt', 'r')) # open all files

    dset_train = np.array(sarsmerscov_train['contact_maps'][:, :, :, 0]).astype(float) # 616207 x 24 x 24
    dset_val = np.array(sarsmerscov_val['contact_maps'][:, :, :, 0]).astype(float) # 152052 x 24 x 24

    return dset_train, dset_val

def rawDataPlotting(training, validating):
    plt.figure(1)
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(training[i, :, :])

    plt.figure(2)
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(validating[i, :, :])

def pca(train, val, numComponents): # TODO: Apply signatures to differentiate clusters
    train_2D = np.resize(train, (train.shape[0], int((train.shape[1] * train.shape[2]) / 2)))  # 616207 x 288
    val_2D = np.resize(val, (val.shape[0], int((val.shape[1] * val.shape[2]) / 2)))  # 152052 x 288

    pca_train = PCA(n_components=numComponents)  # define number of principle components needed
    normalized_train_2D = normalize(train_2D, axis=1, norm='l1') # normalize
    normalized_train_2D = pca_train.fit_transform(normalized_train_2D)
    print(pca_train.explained_variance_ratio_)  # find how much of variance is explained by each component

    pca_val = PCA(n_components=numComponents)
    normalized_val_2D = normalize(val_2D, axis=1, norm='l1') # normalize
    normalized_val_2D = pca_val.fit_transform(normalized_val_2D)
    print(pca_val.explained_variance_ratio_)  # find how much of variance is explained by each component

    return normalized_train_2D, normalized_val_2D

def plotPCA(reduced_t, reduced_v):
    plt.figure(1)
    t_rows = reduced_t[:, 0]
    t_cols = reduced_t[:, 1]
    plt.scatter(t_rows, t_cols)

    plt.figure(2)
    v_rows = reduced_v[:, 0]
    v_cols = reduced_v[:, 1]
    plt.scatter(v_rows, v_cols)

if __name__ == '__main__':
    print("Initializing...")
    trainset, valset = dataLoading()
    print(str(time.ctime()) + ": Successfully loaded all data sets!")

    print(str(time.ctime()) + ": Plotting...")
    rawDataPlotting(trainset, valset)
    print(str(time.ctime()) + ": Finished Plotting!")
    plt.show() # plotting each sample data to see contact maps

    print(str(time.ctime()) + ": Initializing Machine Learing...")
    reduced_train, reduced_val = pca(trainset, valset, 2)
    print(str(time.ctime()) + ": Plotting PCA...")
    plotPCA(reduced_train, reduced_val)
    plt.show() # after PCA clustering plot the first 2 PCs to see clusters
    print(str(time.ctime()) + ": Finished Machine Learning!")
