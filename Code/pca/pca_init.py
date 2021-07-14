import numpy as np
import cupy as cp
import pandas as pd
import time, argparse
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA as sk_PCA
from cuml.decomposition import PCA as cuml_PCA
from dask_ml.decomposition import PCA as cuml_dask_PCA
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import cudf
import dask, dask_cudf

args = argparse.ArgumentParser()
args.add_argument('--npartitions', type=int, help='number of data partitions')
args.add_argument('--single_gpu', type=bool, help='single or multi gpu')
args.add_argument('--dataset', type=str, help='type of data loading in')
args = args.parse_args()
npartitions = args.npartitions
single_gpu = args.single_gpu
datatype = args.dataset

def sklearn_pca(normalized_train_pca, normalized_val_pca):
    print(str(time.ctime()) + ": Implementing PCA Clustering with Sklearn...")

    start = time.time()

    pca = sk_PCA(n_components=2)  # 2 PCs
    pca.fit(normalized_train_pca)
    reduced_train = pca.transform(normalized_train_pca)
    reduced_val = pca.transform(normalized_val_pca) # reduce dimensions of both sets
    print('Total explained variance: {}'.format(pca.explained_variance_ratio_.sum() * 100))

    end = time.time()
    sk_diff = round(end - start, 2)

    print(str(time.ctime()) + ": Finished PCA Clustering with Sklearn in: " + str(sk_diff) + " seconds!")
    return reduced_train, reduced_val
    
def rapids_pca(normalized_train_pca, normalized_val_pca, single_gpu):    
    reduced_train = np.array([])
    reduced_val = np.array([])
        
    if single_gpu:
        print(str(time.ctime()) + ": Transferring CPU->GPU...")
        c_ntpca = cp.array(normalized_train_pca)
        c_nvpca = cp.array(normalized_val_pca)
        print(str(time.ctime()) + ": Successfully transferred!")
        print(str(time.ctime()) + ": Implementing PCA Clustering with Rapids...")
        
        start = time.time()
        
        pca = cuml_PCA(n_components=2)  # 2 PCs
        pca.fit(c_ntpca)
        reduced_train = pca.transform(c_ntpca)
        reduced_val = pca.transform(c_nvpca) # reduce dimensions of both sets
        print('Total explained variance: {}'.format(pca.explained_variance_ratio_.sum() * 100))
    else:
        print(str(time.ctime()) + ": Transferring CPU->GPUs...")
        cluster = LocalCUDACluster(n_workers=npartitions, threads_per_worker=1)
        client  = Client(cluster)
        
        p_ntpca = cudf.DataFrame.from_pandas(pd.DataFrame(normalized_train_pca))
        p_nvpca = cudf.DataFrame.from_pandas(pd.DataFrame(normalized_val_pca))
        
        d_ntpca = dask_cudf.from_cudf(p_ntpca, npartitions=npartitions)
        d_nvpca = dask_cudf.from_cudf(p_nvpca, npartitions=npartitions)
        
        d_ntpca = d_ntpca.persist()
        d_nvpca = d_nvpca.persist()
        print(str(time.ctime()) + ": Successfully transferred!")
        print(str(time.ctime()) + ": Implementing PCA Clustering with Rapids...")
        
        start = time.time()
        
        reduced_train, reduced_val = client.submit(helperMethod, d_ntpca, d_nvpca)
        reduced_train = reduced_train.result()
        reduced_val = reduced_val.result() # use client scheduler to run multi GPU

    end = time.time()
    r_diff = round(end - start, 2)

    print(str(time.ctime()) + ": Finished PCA Clustering with Rapids in: " + str(r_diff) + " seconds!")
    return reduced_train, reduced_val

def helperMethod(dt, dv):
    pca = cuml_dask_PCA(n_components=2)  # 2 PCs
    pca.fit(dt)
    rtrain = cp.array(pca.transform(dt))
    rval = cp.array(pca.transform(dv)) # reduce dimensions of both sets
    print('Total explained variance: {}'.format(pca.explained_variance_ratio_.sum() * 100))
    return rtrain, rval
    
print(str(time.ctime()) + ": Initializing...")
train_pca = None
val_pca = None

if datatype == 'SARSMERSCOV2':
    training = np.array([])
    validation = np.array([])
    
    npzfile = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/sars_mers_cov2_dataset/smc2_dataset.npz')
    training = npzfile['train3D']
    validation = npzfile['val3D']
    
    train_pca = np.reshape(training, (training.shape[0], -1))  # 60000 x 576
    val_pca = np.reshape(validation, (validation.shape[0], -1))  # 15000 x 576
elif datatype == 'HEA':
    npzfile = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/hea_dataset/hea_dataset.npz')
    train_pca = npzfile['train']
    val_pca = npzfile['val']

print(str(time.ctime()) + ": Successfully loaded all data sets!")

normalized_train_pca = normalize(train_pca, axis=1, norm='l1')
normalized_val_pca = normalize(val_pca, axis=1, norm='l1')

# sk_rt, sk_rv = sklearn_pca(normalized_train_pca, normalized_val_pca)
r_rt, r_rv = rapids_pca(normalized_train_pca, normalized_val_pca, single_gpu)

if datatype == 'SARSMERSCOV2':
    # np.savez('smc2_sk_clusterfiles.npz', redtrain=sk_rt, redval=sk_rv)
    cp.savez('smc2_r_clusterfiles.npz', redtrain=r_rt, redval=r_rv)
elif datatype == 'HEA':
    # np.savez('hea_sk_clusterfiles.npz', redtrain=sk_rt, redval=sk_rv)
    cp.savez('hea_r_clusterfiles.npz', redtrain=r_rt, redval=r_rv)