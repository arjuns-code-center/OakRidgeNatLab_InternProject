import numpy as np
import time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA as sk_PCA
# from cuml.decomposition import PCA as cuml_PCA
from sklearn.cluster import KMeans as sk_kmeans
from cuml.cluster import KMeans as cuml_kmeans

npzfile = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/hea_dataset/kmeans/savefile.npz')
trainset = npzfile['train']
trainset = npzfile['val']
label_validation = npzfile['lv']
          
train_pca = np.reshape(trainset, (trainset.shape[0], -1))  # 60000 x 576
val_pca = np.reshape(valset, (valset.shape[0], -1))  # 15000 x 576

normalized_train_pca = normalize(train_pca, axis=1, norm='l1')
normalized_val_pca = normalize(val_pca, axis=1, norm='l1')

print(str(time.ctime()) + ": Implementing PCA Clustering...")

pca = sk_PCA(n_components=2)  # 2 PCs
# pca = cuml_PCA(n_components=2)
pca.fit(normalized_train_pca)
reduced_val = pca.transform(normalized_val_pca) # reduce dimensions of both sets
            
print(str(time.ctime()) + ": Finished PCA Clustering!")

print(str(time.ctime()) + ": Implementing KMeans Clustering with Sklearn...")

start = time.time()
kmeans = sk_kmeans(n_clusters=3, random_state=0)
kmeans.fit(train_pca)
labels_val = np.array(kmeans.predict(val_pca))
accuracy = (sum(labels_val == label_validation) / len(label_validation)) * 100
print('Accuracy: {}'.format(accuracy))

end = time.time()
sk_diff = round(end - start, 2)

print(str(time.ctime()) + ": Finished KMeans Clustering with Sklearn!")

np.savez('sk_plotting.npz', reslab=labels_val, lv=label_validation, redval=reduced_val)

print(str(time.ctime()) + ": Implementing KMeans Clustering with Rapids...")

start = time.time()
kmeans = cuml_kmeans(n_clusters=3, random_state=0)
kmeans.fit(train_pca)
labels_val = np.array(kmeans.predict(val_pca))
accuracy = (sum(labels_val == label_validation) / len(label_validation)) * 100
print('Accuracy: {}'.format(accuracy))

end = time.time()
r_diff = round(end - start, 2)

print(str(time.ctime()) + ": Finished KMeans Clustering with Rapids!")

print("Time for KMeans Clustering with Sklearn: " + str(sk_diff) + " seconds")
print("Time for KMeans Clustering with Rapids: " + str(r_diff) + " seconds")

np.savez('r_plotting.npz', reslab=labels_val, lv=label_validation, redval=reduced_val)