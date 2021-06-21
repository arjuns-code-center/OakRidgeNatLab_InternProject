import h5py
import numpy as np
import time
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, UpSampling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.cm as cmx
from matplotlib import pyplot as plt

print(str(time.ctime()) + ": Initializing...")
sarsmerscov_train = h5py.File('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/cvae/sars-mers-cov2_train.h5', 'r')
sarsmerscov_val = h5py.File('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/cvae/sars-mers-cov2_val.h5', 'r')
lt = list(open('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/label_train.txt', 'r'))
lv = list(open('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/label_val.txt', 'r')) # open all files
cvae_embeddings = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/cvae/sars-mers-cov2-embeddings.npy', 'r')
cvae_samples = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/cvae/sars-mers-cov2-samples.npz', 'r')
        
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

cvae_embeddings = np.squeeze(cvae_embeddings)[0:val_size]

lt = None
lv = None
sarsmerscov_train = None
sarsmerscov_val = None # garbage collection to free up memory

print(str(time.ctime()) + ": Successfully loaded all data sets!")
print(str(time.ctime()) + ": Implementing PCA Clustering...")
            
train_pca = np.reshape(train_3D, (train_3D.shape[0], -1))  # 60000 x 576
val_pca = np.reshape(val_3D, (val_3D.shape[0], -1))  # 15000 x 576

normalized_train_pca = normalize(train_pca, axis=1, norm='l1')
normalized_val_pca = normalize(val_pca, axis=1, norm='l1')

# sc = StandardScaler()
# normalized_train_pca = sc.fit_transform(tpca)
# normalized_val_pca = sc.fit_transform(vpca)

pca = PCA(2)  # 2 PCs
pca.fit(normalized_train_pca)
reduced_train = pca.transform(normalized_train_pca)
reduced_val = pca.transform(normalized_val_pca) # reduce dimensions of both sets
            
print(str(time.ctime()) + ": Finished PCA Clustering!")
print(str(time.ctime()) + ": Creating Classification Model...")

x = Input(shape=(24, 24, 1))  # 24 x 24 x 1

# Encoder
e_conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(x) # 24 x 24 x 8
pool1 = MaxPool2D((2, 2), padding='same')(e_conv1) # 12 x 12 x 8
b_norm1 = BatchNormalization()(pool1)

e_conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(b_norm1) # 12 x 12 x 16
pool2 = MaxPool2D((2, 2), padding='same')(e_conv2) # 6 x 6 x 16
b_norm2 = BatchNormalization()(pool2)

e_conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(b_norm2) # 6 x 6 x 32
pool3 = MaxPool2D((2, 2), padding='same')(e_conv3) # 3 x 3 x 32
b_norm3 = BatchNormalization()(pool3)

e_conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(b_norm3) # 3 x 3 x 64
b_norm4 = BatchNormalization()(e_conv4)

e_conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(b_norm4) # 3 x 3 x 128
b_norm5 = BatchNormalization()(e_conv5)

# Decoder
d_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(b_norm5) # 3 x 3 x 128
up1 = UpSampling2D((2, 2))(d_conv1) # 6 x 6 x 128
b_norm6 = BatchNormalization()(up1)

d_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(b_norm6) # 6 x 6 x 64
up2 = UpSampling2D((2, 2))(d_conv2) # 12 x 12 x 64
b_norm7 = BatchNormalization()(up2)

d_conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(b_norm7) # 12 x 12 x 32
up3 = UpSampling2D((2, 2))(d_conv3) # 24 x 24 x 32
b_norm8 = BatchNormalization()(up3)

d_conv4 = Conv2D(16, (3, 3), activation='relu', padding='same')(b_norm8) # 24 x 24 x 16
b_norm9 = BatchNormalization()(d_conv4)

d_conv5 = Conv2D(8, (3, 3), activation='relu', padding='same')(b_norm9) # 24 x 24 x 8
b_norm10 = BatchNormalization()(d_conv5)

d_conv6 = Conv2D(1, (1, 1), activation='relu', padding='same')(b_norm10) # 24 x 24 x 1
b_norm11 = BatchNormalization()(d_conv6)

f1 = Flatten()(b_norm11) # add a fully connected layer after just the autoencoder. 576 x 1
r = Dense(3, activation='softmax')(f1) # 3 x 1

classification_model = Model(x, r) # compile full model
classification_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['categorical_accuracy'])

print(str(time.ctime()) + ": Successfully created Classification Model")
        
train_X, valid_X, train_label, valid_label = train_test_split(trainset, lt_onehot, test_size=0.2, random_state=0)
# train_X = samples x 24 x 24 x 1
# valid_X = samples x 24 x 24 x 1
# train_label = samples x 3
# valid_label = samples x 3

print(str(time.ctime()) + ": Training Classification Model...")

epochs = 20
batch_size = 64
early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=5, restore_best_weights=True)
classify_labels = classification_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, callbacks=[early_stop], validation_data=(valid_X, valid_label))

print(str(time.ctime()) + ": Finished training!")

print(str(time.ctime()) + ": Evaluating Classification Model...")

test_eval = classification_model.evaluate(valid_X, valid_label)
print('Loss: {}'.format(test_eval[0]))
print('Accuracy: {}'.format(test_eval[1] * 100))

print(str(time.ctime()) + ": Finished evaluation!")

print(str(time.ctime()) + ": Predicting with Classification Model...")

predicted = classification_model.predict(valset)
predicted = np.argmax(np.round(predicted), axis=1)

print(str(time.ctime()) + ": Finished predictions!")

correct = np.where(predicted == label_validation)[0]
incorrect = np.where(predicted != label_validation)[0]
print("Number of Correct Classifications: " + str(len(correct)))
print("Number of Incorrect Classifications: " + str(len(incorrect)))

cNorm = matplotlib.colors.Normalize(vmin=min(predicted), vmax=max(predicted))
scalarMap = cmx.ScalarMappable(norm=cNorm)
fig = plt.figure(7)
plt.scatter(reduced_val[:, 0], reduced_val[:, 1], c=scalarMap.to_rgba(predicted))
scalarMap.set_array(predicted)
cbar = fig.colorbar(scalarMap)
cbar.set_ticks([0,1,2])
cbar.set_ticklabels(["COV2", "MERS", "SARS"])
plt.title('CAE Cluster Map of Predicted Set')
fig.savefig('cae_pred.png')

cNorm = matplotlib.colors.Normalize(vmin=min(label_validation), vmax=max(label_validation))
scalarMap = cmx.ScalarMappable(norm=cNorm)
fig = plt.figure(8)
plt.scatter(reduced_val[:, 0], reduced_val[:, 1], c=scalarMap.to_rgba(label_validation))
scalarMap.set_array(label_validation)
cbar = fig.colorbar(scalarMap)
cbar.set_ticks([0,1,2])
cbar.set_ticklabels(["COV2", "MERS", "SARS"])
plt.title('CAE Cluster Map of Validation Set')
fig.savefig('cae_val.png')