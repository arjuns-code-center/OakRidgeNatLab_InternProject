import h5py
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

sarsmerscov_train = h5py.File('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/cvae/sars-mers-cov2_train.h5', 'r')
sarsmerscov_val = h5py.File('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/cvae/sars-mers-cov2_val.h5', 'r')
lt = list(open('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/label_train.txt', 'r'))
lv = list(open('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/label_val.txt', 'r')) # open all files

label_training = np.array([])
label_validation = np.array([])
        
train_size = 60000
val_size = 15000

for i in range(train_size):
    num = int(str(lt[i]).strip('\n'))
    label_training = np.append(label_training, num)

for j in range(val_size):
    num = int(str(lv[j]).strip('\n'))
    label_validation = np.append(label_validation, num)

trainset = np.array(sarsmerscov_train['contact_maps'][0:train_size]).astype(float) # 60000 x 24 x 24 x 1
valset = np.array(sarsmerscov_val['contact_maps'][0:val_size]).astype(float) # 15000 x 24 x 24 x 1

trainset, label_training = shuffle(trainset, label_training, random_state=0)
valset, label_validation = shuffle(valset, label_validation, random_state=0)

train_3D = np.tril(trainset[:, :, :, 0])
val_3D = np.tril(valset[:, :, :, 0])

lt_onehot = to_categorical(label_training) # make one hot vectors
lv_onehot = to_categorical(label_validation)

lt = None
lv = None
sarsmerscov_train = None
sarsmerscov_val = None # garbage collection to free up memory

np.savez('savefile.npz', train=train_3D, val=val_3D, labval=label_validation, ltoh=lt_onehot, lvoh=lv_onehot)