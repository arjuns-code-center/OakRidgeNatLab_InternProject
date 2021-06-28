import numpy as np
import time, argparse
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans as sk_kmeans
from cuml.cluster import KMeans as cuml_kmeans
from dask_ml.cluster import KMeans as cuml_dask_kmeans
import cudf, dask_cudf
import dask
import pandas as pd

args = argparse.ArgumentParser()
args.add_argument('--npartitions', default=6, type=int, help='number of data partitions')
args.add_argument('--single_gpu', action='store_true', default=False, help='single or multi gpu')
args.add_argument('--dataset', default='SARSMERSCOV2', type=str, help='type of data loading in')
args = args.parse_args()
npartitions = args.npartitions
single_gpu = args.single_gpu
datatype = args.dataset

def sklearn_km(train_pca, val_pca, nclusters, onehotlv):
    print(str(time.ctime()) + ": Implementing KMeans Clustering with Sklearn...")

    start = time.time()
    kmeans = sk_kmeans(n_clusters=nclusters, random_state=0)
    kmeans.fit(train_pca)
    labels_val = np.array(kmeans.predict(val_pca))
    accuracy = (sum(labels_val == onehotlv) / len(onehotlv)) * 100
    print('Accuracy: {}'.format(accuracy))

    end = time.time()
    sk_diff = round(end - start, 2)

    print(str(time.ctime()) + ": Finished KMeans Clustering with Sklearn in: " + str(sk_diff) + " seconds!")

    np.savez('sk_plotting.npz', reslab=labels_val, lv=label_validation, redval=reduced_val)

def rapids_km(train_pca, val_pca, nclusters, single_gpu, onehotlv):
    t_df = pd.DataFrame(train_pca)
    v_df = pd.DataFrame(val_pca)
    
    if single_gpu:
        print(str(time.ctime()) + ": Transferring CPU->GPU...")
        c_tpca = cudf.from_pandas(t_df)
        c_vpca = cudf.from_pandas(v_df)
        print(str(time.ctime()) + ": Successfullu transferred!")
        print(str(time.ctime()) + ": Implementing KMeans Clustering with Rapids...")
        
        start = time.time()
        
        kmeans = cuml_kmeans(n_clusters=nclusters, random_state=0)
        kmeans.fit(c_tpca)
        labels_val = np.array(kmeans.predict(c_vpca))
        accuracy = (sum(labels_val == onehotlv) / len(onehotlv)) * 100
        print('Accuracy: {}'.format(accuracy))
    else:
        print(str(time.ctime()) + ": Transferring CPU->GPU...")
        p_tpca = dask.dataframe.from_pandas(t_df, npartitions=npartitions)
        p_vpca = dask.dataframe.from_pandas(v_df, npartitions=npartitions)
        
        d_tpca = dask_cudf.from_dask_dataframe(p_tpca)
        d_vpca = dask_cudf.from_dask_dataframe(p_vpca)
        
        d_tpca = d_tpca.persist()
        d_vpca = d_vpca.persist()
        print(str(time.ctime()) + ": Successfullu transferred!")
        print(str(time.ctime()) + ": Implementing KMeans Clustering with Rapids...")
        
        start = time.time()
        
        kmeans = cuml_kmeans(n_clusters=nclusters, random_state=0)
        kmeans.fit(d_tpca)
        labels_val = np.array(kmeans.predict(d_vpca))
        accuracy = (sum(labels_val == label_validation) / len(label_validation)) * 100
        print('Accuracy: {}'.format(accuracy))

    end = time.time()
    r_diff = round(end - start, 2)

    print(str(time.ctime()) + ": Finished KMeans Clustering with Rapids in: " + str(r_diff) + " seconds!")

    np.savez('r_plotting.npz', reslab=labels_val, redval=reduced_val, lv=label_validation)
    
training = np.array([])
validation = np.array([])
npzfile = None
nc = 0
if datatype == 'SARSMERSCOV2':
    npzfile = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/sars_mers_cov2_dataset/smc2_dataset.npz')
    training = npzfile['train3D']
    valiidation = npzfile['val3D']
    nc = 3
elif datatype == 'HEA':
    npzfile = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/hea_dataset/hea_dataset.npz')
    training = npzfile['train']
    valiidation = npzfile['val']
    nc = 5
    
label_validation = npzfile['label_validation']
lv_onehot = npzfile['lvoh']
l = np.array([]).astype(int)
for i in range(len(lv_onehot)):
    l = np.append(l, np.argmax(lv_onehot[i]))

train_pca = np.reshape(training, (training.shape[0], -1)) 
val_pca = np.reshape(validation, (validation.shape[0], -1)) 

sklearn_km(train_pca, val_pca, nc, lvoh)
rapids_km(train_pca, val_pca, nc, single_gpu, lvoh)