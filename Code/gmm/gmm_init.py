import numpy as np
import time, argparse
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture as sk_gmm
from sklearn.decomposition import PCA as sk_PCA
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential
import tensorflow_probability as tfp

args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, help='type of data loading in')
args = args.parse_args()
datatype = args.dataset

def implementPCA(train, val):
    print(str(time.ctime()) + ": Implementing PCA Clustering with Sklearn...")

    pca = sk_PCA(n_components=2)  # 2 PCs
    pca.fit(train)
    train = pca.transform(train)
    val = pca.transform(val) # reduce dimensions of both sets
    print('Total explained variance: {}'.format(pca.explained_variance_ratio_.sum() * 100))

    print(str(time.ctime()) + ": Finished PCA Clustering with Sklearn!")
    return train, val

def sklearn_gmm(normalized_train_pca, normalized_val_pca, nc):   
    reduced_train, reduced_val = implementPCA(normalized_train_pca, normalized_val_pca)
    
    print(str(time.ctime()) + ": Implementing GMM Clustering with Sklearn...")
    start = time.time()
    
    gmm = sk_gmm(n_components=nc, covariance_type='full')
    gmm.fit(reduced_train)
    gmm_predicted = gmm.predict(reduced_val)
    
    end = time.time()
    sk_diff = round(end - start, 2)
    
    print(str(time.ctime()) + ": Finished GMM Clustering with Sklearn in " + str(sk_diff) + " seconds!")
    return gmm_predicted, reduced_val

def tensorflow_gmm(normalized_train_pca, normalized_val_pca, lt, lv, nc):
    reduced_train, reduced_val = implementPCA(normalized_train_pca, normalized_val_pca)
    event_shape = [nc]
    params_size = tfp.layers.MixtureNormal.params_size(nc, event_shape)
    
    print(str(time.ctime()) + ": Creating Model...")
    
    model = Sequential()
    model.add(Dense(16, activation='relu'))
    model.add(Dense(params_size, activation=None))
    model.add(tfp.layers.MixtureNormal(nc, event_shape))
    
    batch_size = 100
    epochs = 10
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=5, restore_best_weights=True)
    print(str(time.ctime()) + ": Model Created Successfully!")
    
    print(str(time.ctime()) + ": Implementing GMM Clustering with Tensorflow...")
    start = time.time()
    
    history = model.fit(reduced_train, lt, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=early_stop, validation_data=(reduced_val, lv))
    pred_labels = model.predict(reduced_val)
    pred_labels = np.argmax(np.round(pred_labels), axis=1)
    
    end = time.time()
    
    tf_diff = round(end - start, 2)
    
    print(str(time.ctime()) + ": Finished GMM Clustering with Tensorflow in " + str(tf_diff) + " seconds!")
    return pred_labels, reduced_val
        
print(str(time.ctime()) + ": Initializing...")
train_pca = None
val_pca = None
nclusters = 0

if datatype == 'SARSMERSCOV2':
    training = np.array([])
    validation = np.array([])

    npzfile = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/sars_mers_cov2_dataset/smc2_dataset.npz')
    training = npzfile['train3D']
    validation = npzfile['val3D']

    train_pca = np.reshape(training, (training.shape[0], -1))  # 60000 x 576
    val_pca = np.reshape(validation, (validation.shape[0], -1))  # 15000 x 576
    
    nclusters = 3
elif datatype == 'HEA':
    npzfile = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/hea_dataset/hea_dataset.npz')
    train_pca = npzfile['train']
    val_pca = npzfile['val']
    
    nclusters = 5
    
label_validation = npzfile['labval']
lt_onehot = npzfile['ltoh']
lv_onehot = npzfile['lvoh']
l = np.array([]).astype(int)
for i in range(len(lv_onehot)):
    l = np.append(l, np.argmax(lv_onehot[i]))

print(str(time.ctime()) + ": Successfully loaded all data sets!")

normalized_train_pca = normalize(train_pca, axis=1, norm='l1')
normalized_val_pca = normalize(val_pca, axis=1, norm='l1')

# sk_gmm_labels, sk_rv = sklearn_gmm(normalized_train_pca, normalized_val_pca, nclusters)
# accuracy = (sum(sk_gmm_labels == l) / len(l)) * 100
# print('Accuracy: {}'.format(accuracy))

tf_gmm_labels, tf_rv = tensorflow_gmm(normalized_train_pca, normalized_val_pca, lt_onehot, lv_onehot, nclusters)
accuracy = (sum(tf_gmm_labels == l) / len(l)) * 100
print('Accuracy: {}'.format(accuracy))

if datatype == 'SARSMERSCOV2':
    #np.savez('smc2_sk_clusterfiles.npz', redval=sk_rv, pred_labels=sk_gmm_labels, lv=label_validation, onehotmax=l)
    np.savez('smc2_tf_clusterfiles.npz', redval=tf_rv, pred_labels=tf_gmm_labels, lv=label_validation, onehotmax=l)
elif datatype == 'HEA':
    #np.savez('hea_sk_clusterfiles.npz', redval=sk_rv, pred_labels=sk_gmm_labels, lv=label_validation, onehotmax=l)
    np.savez('hea_tf_clusterfiles.npz', redval=tf_rv, pred_labels=tf_gmm_labels,lv=label_validation, onehotmax=l)