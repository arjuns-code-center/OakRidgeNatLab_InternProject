import numpy as np
import time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA as sk_PCA
# from cuml.decomposition import PCA as cuml_PCA
# from cuml.dask.decomposition import PCA

print(str(time.ctime()) + ": Implementing PCA Clustering...")

npzfile = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/sars_mers_covid_dataset/cae/savefile.npz')
train_3D = npzfile['train3D']
val_3D = npzfile['val3D']

train_pca = np.reshape(train_3D, (train_3D.shape[0], -1))  # 60000 x 576
val_pca = np.reshape(val_3D, (val_3D.shape[0], -1))  # 15000 x 576

normalized_train_pca = normalize(train_pca, axis=1, norm='l1')
normalized_val_pca = normalize(val_pca, axis=1, norm='l1')

pca = sk_PCA(n_components=2)  # 2 PCs
# pca = cuml_PCA(n_components=2)
pca.fit(normalized_train_pca)
reduced_train = pca.transform(normalized_train_pca)
reduced_val = pca.transform(normalized_val_pca) # reduce dimensions of both sets
            
print(str(time.ctime()) + ": Finished PCA Clustering!")

np.savez('clusterfiles.npz', redtrain=reduced_train, redval=reduced_val)