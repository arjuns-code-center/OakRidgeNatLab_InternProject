import numpy as np
import time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA as sk_PCA
from cuml.decomposition import PCA as cuml_PCA
# from cuml.dask.decomposition import PCA

print(str(time.ctime()) + ": Initializing...")

npzfile = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/pca/savefile.npz')
train_3D = npzfile['train']
val_3D = npzfile['val']

print(str(time.ctime()) + ": Successfully loaded all data sets!")
            
train_pca = np.reshape(train_3D, (train_3D.shape[0], -1))  # 60000 x 576
val_pca = np.reshape(val_3D, (val_3D.shape[0], -1))  # 15000 x 576

normalized_train_pca = normalize(train_pca, axis=1, norm='l1')
normalized_val_pca = normalize(val_pca, axis=1, norm='l1')

print(str(time.ctime()) + ": Implementing PCA Clustering with Sklearn...")

start = time.time()

pca = sk_PCA(n_components=2)  # 2 PCs
pca.fit(normalized_train_pca)
reduced_train = pca.transform(normalized_train_pca)
reduced_val = pca.transform(normalized_val_pca) # reduce dimensions of both sets
print('Total explained variance: {}'.format(pca.explained_variance_ratio_.sum() * 100))

np.savez('sk_clusterfiles.npz', redtrain=reduced_train, redval=reduced_val)

end = time.time()
sk_diff = round(end - start, 2)

print(str(time.ctime()) + ": Finished PCA Clustering with Sklearn!")
print(str(time.ctime()) + ": Implementing PCA Clustering with Rapids...")

start = time.time()

pca = cuml_PCA(n_components=2)  # 2 PCs
pca.fit(normalized_train_pca)
reduced_train = pca.transform(normalized_train_pca)
reduced_val = pca.transform(normalized_val_pca) # reduce dimensions of both sets
print('Total explained variance: {}'.format(pca.explained_variance_ratio_.sum() * 100))

np.savez('r_clusterfiles.npz', redtrain=reduced_train, redval=reduced_val)

end = time.time()
r_diff = round(end - start, 2)

print(str(time.ctime()) + ": Finished PCA Clustering with Rapids!")

print("Time for Sklearn PCA Clustering: " + str(sk_diff) + " seconds")
print("Time for Rapids PCA Clustering: " + str(r_diff) + " seconds")