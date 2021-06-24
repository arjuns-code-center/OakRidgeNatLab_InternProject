import numpy as np
import time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA as sk_PCA
from cuml.decomposition import PCA as cuml_PCA
# from cuml.dask.decomposition import PCA

print(str(time.ctime()) + ": Initializing...")

npzfile = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/hea_dataset/pca/savefile.npz')
trainset = npzfile['train']
valset = npzfile['val']

print(str(time.ctime()) + ": Successfully loaded all data sets!")
            
train_pca = np.reshape(trainset, (trainset.shape[0], -1))  # 40000 x 442368
val_pca = np.reshape(valset, (valset.shape[0], -1))  # 10000 x 442368

normalized_train_pca = normalize(train_pca, axis=1, norm='l1')
normalized_val_pca = normalize(val_pca, axis=1, norm='l1')

print(str(time.ctime()) + ": Implementing PCA Clustering...")

pca = sk_PCA(n_components=2)  # 2 PCs
# pca = cuml_PCA(n_components=2)
pca.fit(normalized_train_pca)
reduced_train = pca.transform(normalized_train_pca)
reduced_val = pca.transform(normalized_val_pca) # reduce dimensions of both sets
print('Total explained variance: {}'.format(pca.explained_variance_ratio_.sum() * 100))

np.savez('clusterfiles.npz', redtrain=reduced_train, redval=reduced_val)

print(str(time.ctime()) + ": Finished PCA Clustering!")