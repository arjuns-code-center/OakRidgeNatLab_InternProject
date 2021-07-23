import numpy as np
import cupy as cp
import pandas as pd
import time, argparse
from sklearn.cluster import KMeans as sk_kmeans
from cuml.cluster import KMeans as cuml_kmeans
from cuml.dask.cluster import KMeans as dask_kmeans
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import cudf, dask_cudf, dask

args = argparse.ArgumentParser()
args.add_argument('--npartitions', type=int, help='number of data partitions')
args.add_argument('--single_gpu', type=bool, help='single or multi gpu')
args.add_argument('--dataset', type=str, help='type of data loading in')
args = args.parse_args()
npartitions = args.npartitions
single_gpu = args.single_gpu
datatype = args.dataset

def sklearn_km(train_pca, val_pca, nclusters):
    print(str(time.ctime()) + ": Implementing KMeans Clustering with Sklearn...")
    start = time.time()
    
    kmeans = sk_kmeans(n_clusters=nclusters, random_state=0)
    kmeans.fit(train_pca)
    labels_val = np.array(kmeans.predict(val_pca))
    
    end = time.time()
    sk_diff = round(end - start, 2)
    print(str(time.ctime()) + ": Finished KMeans Clustering with Sklearn in: " + str(sk_diff) + " seconds!")
    
    return labels_val

def rapids_km(train_pca, val_pca, nclusters, single_gpu):
    kmeans = None
    name = None
    
    if single_gpu:
        print(str(time.ctime()) + ": Transferring CPU->GPU...")
        train_pca = cp.array(train_pca)
        val_pca = cp.array(val_pca)
        print(str(time.ctime()) + ": Successfully Transferred!")

        name = "Rapids"
        kmeans = cuml_kmeans(n_clusters=nclusters, random_state=0)
    else:
        print(str(time.ctime()) + ": Transferring CPU->GPUs...")
        train_pca = dask_cudf.from_dask_dataframe(dask.dataframe.from_pandas(pd.DataFrame(train_pca)), npartitions=npartitions).persist()
        val_pca = dask_cudf.from_dask_dataframe(dask.dataframe.from_pandas(pd.DataFrame(val_pca)), npartitions=npartitions).persist()
        print(str(time.ctime()) + ": Successfully Transferred!")

        cluster = LocalCUDACluster(n_workers=npartitions, threads_per_worker=1)
        client  = Client(cluster)

        name = "Dask"
        kmeans = dask_kmeans(n_clusters=nclusters, random_state=0)
        
    print(str(time.ctime()) + ": Implementing KMeans Clustering with " + name + "...")
    kmeans.fit(train_pca)
    runtimes = np.array([])
    
    for i in range(10):
        start = time.time()
        labels_val = cp.asnumpy(cp.array(kmeans.predict(val_pca)))
        end = time.time()
        r_diff = end - start
        
        runtimes = np.append(runtimes, r_diff)
    
    avg_runtime = round(sum(runtimes[3:]) / len(runtimes[3:]), 2)
    print(str(time.ctime()) + ": Finished KMeans Clustering with " + name + " in: " + str(avg_runtime) + " seconds!")
    return labels_val
    
train_pca = None
val_pca = None
npzfile = None
nc = 0
if datatype == 'SARSMERSCOV2':    
    npzfile = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/sars_mers_cov2_dataset/smc2_dataset.npz')
    training = npzfile['train3D']
    validation = npzfile['val3D']
    nc = 3
    
    train_pca = np.reshape(training, (training.shape[0], -1)) 
    val_pca = np.reshape(validation, (validation.shape[0], -1)) 
elif datatype == 'HEA':
    npzfile = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/hea_dataset/hea_dataset.npz')
    train_pca = npzfile['train']
    val_pca = npzfile['val']
    nc = 5
    
label_validation = npzfile['labval']
lv_onehot = npzfile['lvoh']
l = np.array([]).astype(int)
for i in range(len(lv_onehot)):
    l = np.append(l, np.argmax(lv_onehot[i]))

sk_labels_val = sklearn_km(train_pca, val_pca, nc)
accuracy = (sum(sk_labels_val == l) / len(l)) * 100
print('Accuracy: {}'.format(accuracy))

r_labels_val = rapids_km(train_pca, val_pca, nc, single_gpu)
accuracy = (sum(r_labels_val == l) / len(l)) * 100
print('Accuracy: {}'.format(accuracy))

if datatype == 'SARSMERSCOV2':
    npzfile2 = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/pca/smc2_sk_clusterfiles.npz')
    reduced_val = npzfile2['redval']
    np.savez('smc2_sk_plotting.npz', reslab=sk_labels_val, lv=label_validation, redval=reduced_val)
    np.savez('smc2_r_plotting.npz', reslab=r_labels_val, redval=reduced_val, lv=label_validation)
elif datatype == 'HEA':
    npzfile2 = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/pca/hea_sk_clusterfiles.npz')
    reduced_val = npzfile2['redval']
    np.savez('hea_sk_plotting.npz', reslab=sk_labels_val, lv=label_validation, redval=reduced_val)
    np.savez('hea_r_plotting.npz', reslab=r_labels_val, redval=reduced_val, lv=label_validation)