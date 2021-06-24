import numpy as np
import time
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

print(str(time.ctime()) + ": Initializing...")
trainset = np.load('/gpfs/alpine/gen150/proj-shared/junqi/hea/HEA_train.npy')
valset = np.load('/gpfs/alpine/gen150/proj-shared/junqi/hea/HEA_val.npy')
label_training = np.load('/gpfs/alpine/gen150/proj-shared/junqi/hea/label_T_train.npy')
label_validation = np.load('/gpfs/alpine/gen150/proj-shared/junqi/hea/label_T_val.npy')

print(trainset.shape)
print(valset.shape)
print(label_training.shape)
print(label_validation.shape)

trainset, label_training = shuffle(trainset, label_training, random_state=0)
valset, label_validation = shuffle(valset, label_validation, random_state=0)
        
# train_size = 40000
# val_size = 10000
# trainset = trainset[0:train_size]
# valset = valset[0:val_size]
# label_training = label_training[0:train_size]
# label_validation = label_validation[0:val_size]

lt_onehot = to_categorical(label_training) # make one hot vectors
lv_onehot = to_categorical(label_validation)

np.savez('savefile.npz', train=trainset, val=valset, labval=label_validation, ltoh=lt_onehot, lvoh=lv_onehot)

print(str(time.ctime()) + ": Successfully loaded all data sets!")