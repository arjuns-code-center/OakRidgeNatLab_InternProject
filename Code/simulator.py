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
from mpl_toolkits import mplot3d
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential

def dataLoading():
    sarsmerscov_train = h5py.File('D:\\ORNL_Code_Data\\sars-mers-cov2_train.h5', 'r')
    sarsmerscov_val = h5py.File('D:\\ORNL_Code_Data\\sars-mers-cov2_val.h5', 'r')
    label_training = list(open('D:\\ORNL_Coding\\Data Files\\label_train.txt', 'r'))
    label_validation = list(open('D:\\ORNL_Coding\\Data Files\\label_val.txt', 'r')) # open all files

    dset_train = np.array(sarsmerscov_train['contact_maps'][:, :, :, 0]).astype(float) # 616207 x 24 x 24
    dset_val = np.array(sarsmerscov_val['contact_maps'][:, :, :, 0]).astype(float) # 152052 x 24 x 24

    return dset_train, dset_val, label_training, label_validation

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

def pca(train, val, numComponents):
    train_pca = np.resize(train, (train.shape[0], int((train.shape[1] * train.shape[2]) / 2)))  # 616207 x 288
    val_pca = np.resize(val, (val.shape[0], int((val.shape[1] * val.shape[2]) / 2)))  # 152052 x 288

    pca_train = PCA(n_components=numComponents)  # define number of principle components needed
    normalized_train_pca = normalize(train_pca, axis=1, norm='l1') # normalize
    normalized_train_pca = pca_train.fit_transform(normalized_train_pca)
    # print(pca_train.explained_variance_ratio_)  # find how much of variance is explained by each component

    pca_val = PCA(n_components=numComponents)
    normalized_val_pca = normalize(val_pca, axis=1, norm='l1') # normalize
    normalized_val_pca = pca_val.fit_transform(normalized_val_pca)
    # print(pca_val.explained_variance_ratio_)  # find how much of variance is explained by each component

    return normalized_train_pca, normalized_val_pca

def plotPCA_2D(reduced_t, reduced_v, txt_t, txt_v):
    plt.figure(1)
    t_rows_sars = np.array([])
    t_cols_sars = np.array([])
    t_rows_mers = np.array([])
    t_cols_mers = np.array([])
    t_rows_covid = np.array([])
    t_cols_covid = np.array([])
    for sample in range(reduced_t.shape[0]):
        num = int(str(txt_t[sample]).strip('\n'))
        if num == 0:
            t_rows_covid = np.append(t_rows_covid, reduced_t[sample, 0])
            t_cols_covid = np.append(t_cols_covid, reduced_t[sample, 1])
        elif num == 1:
            t_rows_mers = np.append(t_rows_mers, reduced_t[sample, 0])
            t_cols_mers = np.append(t_cols_mers, reduced_t[sample, 1])
        elif num == 2:
            t_rows_sars = np.append(t_rows_sars, reduced_t[sample, 0])
            t_cols_sars = np.append(t_cols_sars, reduced_t[sample, 1])
    plt.scatter(t_rows_sars, t_cols_sars, c='b', label='SARS')
    plt.scatter(t_rows_mers, t_cols_mers, c='r', label='MERS')
    plt.scatter(t_rows_covid, t_cols_covid, c='k', label='COVID')
    plt.legend(loc='upper right')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Scatter plot showing PCA clustering for training dataset')

    plt.figure(2)
    v_rows_sars = np.array([])
    v_cols_sars = np.array([])
    v_rows_mers = np.array([])
    v_cols_mers = np.array([])
    v_rows_covid = np.array([])
    v_cols_covid = np.array([])
    for sample in range(reduced_v.shape[0]):
        num = int(str(txt_v[sample]).strip('\n'))
        if num == 0:
            v_rows_covid = np.append(v_rows_covid, reduced_v[sample, 0])
            v_cols_covid = np.append(v_cols_covid, reduced_v[sample, 1])
        elif num == 1:
            v_rows_mers = np.append(v_rows_mers, reduced_v[sample, 0])
            v_cols_mers = np.append(v_cols_mers, reduced_v[sample, 1])
        elif num == 2:
            v_rows_sars = np.append(v_rows_sars, reduced_v[sample, 0])
            v_cols_sars = np.append(v_cols_sars, reduced_v[sample, 1])
    plt.scatter(v_rows_sars, v_cols_sars, c='b', label='SARS')
    plt.scatter(v_rows_mers, v_cols_mers, c='r', label='MERS')
    plt.scatter(v_rows_covid, v_cols_covid, c='k', label='COVID')
    plt.legend(loc='upper right')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Scatter plot showing PCA clustering for validation dataset')

def plotPCA_3D(reduced_t, reduced_v, txt_t, txt_v):
    plt.figure(3)
    ax = plt.axes(projection='3d')
    t_x_sars = np.array([])
    t_y_sars = np.array([])
    t_z_sars = np.array([])
    t_x_mers = np.array([])
    t_y_mers = np.array([])
    t_z_mers = np.array([])
    t_x_covid = np.array([])
    t_y_covid = np.array([])
    t_z_covid = np.array([])
    for sample in range(reduced_t.shape[0]):
        num = int(str(txt_t[sample]).strip('\n'))
        if num == 0:
            t_x_covid = np.append(t_x_covid, reduced_t[sample, 0])
            t_y_covid = np.append(t_y_covid, reduced_t[sample, 1])
            t_z_covid = np.append(t_z_covid, reduced_t[sample, 2])
        elif num == 1:
            t_x_mers = np.append(t_x_mers, reduced_t[sample, 0])
            t_y_mers = np.append(t_y_mers, reduced_t[sample, 1])
            t_z_mers = np.append(t_z_mers, reduced_t[sample, 2])
        elif num == 2:
            t_x_sars = np.append(t_x_sars, reduced_t[sample, 0])
            t_y_sars = np.append(t_y_sars, reduced_t[sample, 1])
            t_z_sars = np.append(t_z_sars, reduced_t[sample, 2])
    ax.scatter3D(t_x_sars, t_y_sars, t_z_sars, c='b', label='SARS')
    ax.scatter3D(t_x_mers, t_y_mers, t_z_mers, c='r', label='MERS')
    ax.scatter3D(t_x_covid, t_y_covid, t_z_covid, c='k', label='COVID')
    plt.legend(loc='upper right')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.title('Scatter plot showing PCA clustering for training dataset')

    plt.figure(4)
    ax2 = plt.axes(projection='3d')
    v_x_sars = np.array([])
    v_y_sars = np.array([])
    v_z_sars = np.array([])
    v_x_mers = np.array([])
    v_y_mers = np.array([])
    v_z_mers = np.array([])
    v_x_covid = np.array([])
    v_y_covid = np.array([])
    v_z_covid = np.array([])
    for sample in range(reduced_v.shape[0]):
        num = int(str(txt_v[sample]).strip('\n'))
        if num == 0:
            v_x_covid = np.append(v_x_covid, reduced_v[sample, 0])
            v_y_covid = np.append(v_y_covid, reduced_v[sample, 1])
            v_z_covid = np.append(v_z_covid, reduced_v[sample, 2])
        elif num == 1:
            v_x_mers = np.append(v_x_mers, reduced_v[sample, 0])
            v_y_mers = np.append(v_y_mers, reduced_v[sample, 1])
            v_z_mers = np.append(v_z_mers, reduced_v[sample, 2])
        elif num == 2:
            v_x_sars = np.append(v_x_sars, reduced_v[sample, 0])
            v_y_sars = np.append(v_y_sars, reduced_v[sample, 1])
            v_z_sars = np.append(v_z_sars, reduced_v[sample, 2])
    ax2.scatter3D(v_x_sars, v_y_sars, v_z_sars, c='b', label='SARS')
    ax2.scatter3D(v_x_mers, v_y_mers, v_z_mers, c='r', label='MERS')
    ax2.scatter3D(v_x_covid, v_y_covid, v_z_covid, c='k', label='COVID')
    plt.legend(loc='upper right')
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.set_zlabel('Principal Component 3')
    plt.title('Scatter plot showing PCA clustering for training dataset')

if __name__ == '__main__':
    print(str(time.ctime()) + ": Initializing...")
    trainset, valset, traintxt, valtxt = dataLoading()
    print(str(time.ctime()) + ": Successfully loaded all data sets!")

    print(str(time.ctime()) + ": Plotting contact maps...")
    rawDataPlotting(trainset, valset)
    print(str(time.ctime()) + ": Finished Plotting!")
    plt.show() # plotting each sample data to see contact maps

    print(str(time.ctime()) + ": Implementing PCA Clustering...")
    reduced_train_2D, reduced_val_2D = pca(trainset, valset, 2)
    reduced_train_3D, reduced_val_3D = pca(trainset, valset, 3)
    print(str(time.ctime()) + ": Finished PCA Clustering!")

    print(str(time.ctime()) + ": Plotting PCA...")
    plotPCA_2D(reduced_train_2D, reduced_val_2D, traintxt, valtxt)
    plotPCA_3D(reduced_train_3D, reduced_val_3D, traintxt, valtxt)
    print(str(time.ctime()) + ": Finished PCA Plotting!")
    plt.show() # after PCA clustering plot the first n PCs to see clusters

    print(str(time.ctime()) + ": Implementing Machine Learning...")
    epochs = 20
    batch_size = 128
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(reduced_train_2D.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))

    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])
    history = model.fit(reduced_train_2D, batch_size=batch_size, epochs=epochs, validation_data=reduced_val_2D)
    print(str(time.ctime()) + ": Finished Machine Learning!")