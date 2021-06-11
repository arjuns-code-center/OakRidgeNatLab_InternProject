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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from mpl_toolkits import mplot3d
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model

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

    sc = StandardScaler()
    sc.fit(train_pca) # fit the scaler to the validation set
    normalized_train_pca = sc.transform(train_pca)
    normalized_val_pca = sc.transform(val_pca)  # normalize both sets

    pca = PCA(n_components=numComponents)  # define number of principle components needed
    pca.fit(normalized_train_pca) # fit pca to validation set
    normalized_train_pca = pca.transform(normalized_train_pca)
    normalized_val_pca = pca.transform(normalized_val_pca) # reduce dimensions of both sets
    # print(pca.explained_variance_ratio_)  # find how much of variance is explained by each component

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
    plt.scatter(t_rows_sars, t_cols_sars, c='b', label='SARS', alpha=1)
    plt.scatter(t_rows_mers, t_cols_mers, c='r', label='MERS', alpha=1)
    plt.scatter(t_rows_covid, t_cols_covid, c='g', label='COVID', alpha=1)
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
    plt.scatter(v_rows_sars, v_cols_sars, c='b', label='SARS', alpha=1)
    plt.scatter(v_rows_mers, v_cols_mers, c='r', label='MERS', alpha=1)
    plt.scatter(v_rows_covid, v_cols_covid, c='g', label='COVID', alpha=1)
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
    ax.scatter3D(t_x_sars, t_y_sars, t_z_sars, c='b', label='SARS', alpha=1)
    ax.scatter3D(t_x_mers, t_y_mers, t_z_mers, c='r', label='MERS', alpha=1)
    ax.scatter3D(t_x_covid, t_y_covid, t_z_covid, c='g', label='COVID', alpha=1)
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
    ax2.scatter3D(v_x_sars, v_y_sars, v_z_sars, c='b', label='SARS', alpha=1)
    ax2.scatter3D(v_x_mers, v_y_mers, v_z_mers, c='r', label='MERS', alpha=1)
    ax2.scatter3D(v_x_covid, v_y_covid, v_z_covid, c='g', label='COVID', alpha=1)
    plt.legend(loc='upper right')
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.set_zlabel('Principal Component 3')
    plt.title('Scatter plot showing PCA clustering for validation dataset')

def kmeans(train, val):
    km = KMeans(n_clusters=3, random_state=0)
    labels_train = np.array(km.fit_predict(train))
    labels_val = np.array(km.fit_predict(val))
    return labels_train, labels_val

def plotKMeans_2D(train, val, t_labels, v_labels):
    t_label0 = train[np.where(t_labels == 0)]
    t_label1 = train[np.where(t_labels == 1)]
    t_label2 = train[np.where(t_labels == 2)]

    v_label0 = val[np.where(v_labels == 0)]
    v_label1 = val[np.where(v_labels == 1)]
    v_label2 = val[np.where(v_labels == 2)]

    plt.figure(1)
    plt.scatter(t_label0[:, 0], t_label0[:, 1], c='b', label='COVID')
    plt.scatter(t_label1[:, 0], t_label1[:, 1], c='r', label='MERS')
    plt.scatter(t_label2[:, 0], t_label2[:, 1], c='g', label='SARS')
    plt.title('K-Means Cluster Map of Training Set')
    plt.legend(loc='upper right')

    plt.figure(2)
    plt.scatter(v_label0[:, 0], v_label0[:, 1], c='b', label='COVID')
    plt.scatter(v_label1[:, 0], v_label1[:, 1], c='r', label='MERS')
    plt.scatter(v_label2[:, 0], v_label2[:, 1], c='g', label='SARS')
    plt.title('K-Means Cluster Map of Validation Set')
    plt.legend(loc='upper right')

def create_model(xdim, ydim, zdim): # base convolutional autoencoder
    x = Input(shape=(xdim, ydim, zdim))  # 24 x 24 x 1
    e_conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # 24 x 24 x 32
    pool1 = MaxPool2D((2, 2), padding='same')(e_conv1)  # 12 x 12 x 32

    e_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # 12 x 12 x 64
    pool2 = MaxPool2D((2, 2), padding='same')(e_conv2)  # 6 x 6 x 64

    e_conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)  # 6 x 6 x 128

    # Decoder - reconstructs the input from a latent representation
    d_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(e_conv3)  # 6 x 6 x 128
    up1 = UpSampling2D((2, 2))(d_conv1)  # 12 x 12 x 128

    d_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)  # 12 x 12 x 64
    up2 = UpSampling2D((2, 2))(d_conv2)  # 24 x 24 x 64

    r = Conv2D(1, (1, 1), activation='sigmoid')(up2)  # 22 x 22 x 1

    model = Model(x, r)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    return model

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

    print(str(time.ctime()) + ": Implementing K-Means Clustering...")
    l_t, l_v = kmeans(reduced_val_2D, reduced_val_2D)
    print(str(time.ctime()) + ": Finished K-Means Clustering!")

    print(str(time.ctime()) + ": Plotting K-Means...")
    plotKMeans_2D(reduced_train_2D, reduced_val_2D, l_t, l_v)
    print(str(time.ctime()) + ": Finished K-Means Plotting!")
    plt.show()

    print(str(time.ctime()) + ": Implementing Machine Learning...")
    epochs = 20
    batch_size = 64

    X_train, X_valid, y_train, y_valid = train_test_split(trainset, trainset, test_size=0.2, random_state=13)

    autoencoder = create_model(24, 24, 1)
    # print(autoencoder.summary())
    history = autoencoder.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    result = autoencoder.predict(valset)
    loss_val = autoencoder.evaluate(result, valset)
    print("Loss: " + str(loss_val))

    print(str(time.ctime()) + ": Finished Machine Learning!")