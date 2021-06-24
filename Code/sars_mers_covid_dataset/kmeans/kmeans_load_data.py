import h5py
import numpy as np
import time
from sklearn.utils import shuffle

print(str(time.ctime()) + ": Initializing...")
sarsmerscov_train = h5py.File('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/sars_mers_covid_dataset/cvae/sars-mers-cov2_train.h5', 'r')
sarsmerscov_val = h5py.File('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/sars_mers_covid_dataset/cvae/sars-mers-cov2_val.h5', 'r')
lt = list(open('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/sars_mers_covid_dataset/label_train.txt', 'r'))
lv = list(open('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/sars_mers_covid_dataset/label_val.txt', 'r')) # open all files

label_training = np.array([])
label_validation = np.array([])

for i in range(len(lt)):
    num = int(str(lt[i]).strip('\n'))
    label_training = np.append(label_training, num)

for j in range(len(lv)):
    num = int(str(lv[j]).strip('\n'))
    label_validation = np.append(label_validation, num)

trainset = np.array(sarsmerscov_train['contact_maps']).astype(float) # samples x 24 x 24 x 1
valset = np.array(sarsmerscov_val['contact_maps']).astype(float) # samples x 24 x 24 x 1

trainset, label_training = shuffle(trainset, label_training, random_state=0)
valset, label_validation = shuffle(valset, label_validation, random_state=0)

train_size = 60000
val_size = 15000
trainset = trainset[0:train_size]
valset = valset[0:val_size]
label_training = label_training[0:train_size]
label_validation = label_validation[0:val_size]

train_3D = np.tril(trainset[:, :, :, 0])
val_3D = np.tril(valset[:, :, :, 0])

lt = None
lv = None
sarsmerscov_train = None
sarsmerscov_val = None # garbage collection to free up memory

np.savez('savefile.npz', train=train_3D, val=val_3D, lv=label_validation)

print(str(time.ctime()) + ": Successfully loaded all data sets!")
