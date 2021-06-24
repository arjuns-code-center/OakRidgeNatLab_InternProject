import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import horovod.tensorflow.keras as hvd

print(str(time.ctime()) + ": Implementing PCA ML...")

npzfile1 = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/hea_dataset/pca/sk_clusterfiles.npz')
npzfile2 = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/hea_dataset/pca/savefile.npz')
reduced_train = npzfile1['redtrain']
reduced_val = npzfile1['redval']
lt_onehot = npzfile2['ltoh']
lv_onehot = npzfile2['lvoh']
label_validation = npzfile2['labval']

hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        
pcamodel = Sequential()
pcamodel.add(Dense(128, activation='relu', input_shape=(2,)))
pcamodel.add(Dense(64, activation='relu'))
pcamodel.add(Dense(64, activation='relu'))
pcamodel.add(Dense(32, activation='relu'))
pcamodel.add(Dense(3, activation='softmax'))

opt = tf.keras.optimizers.Adam(0.001 * hvd.size()) # adjust learning rate based on # GPUs
opt = hvd.DistributedOptimizer(opt) # add distributed optimizer
pcamodel.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
        
batch_size = 64
epochs = 20
# early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=5, restore_best_weights=True)
callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

# if hvd.rank() == 0:
#     callbacks.append(tf.keras.callbacks.ModelCheckpoint('./best_checkpoint-{epoch}.h5', monitor='val_categorical_accuracy', mode='max', save_best_only=True))

history = pcamodel.fit(reduced_train, lt_onehot, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(reduced_val, lv_onehot), callbacks=callbacks)
        
print(str(time.ctime()) + ": Finished PCA ML")
print(str(time.ctime()) + ": Predicting with PCA Model...")

result = pcamodel.predict(reduced_val)
result = np.argmax(np.round(result), axis=1)
        
print(str(time.ctime()) + ": Finished predictions!")

correct = np.where(result == label_validation)[0]
incorrect = np.where(result != label_validation)[0]
print("Number of Correct Classifications: {}".format(len(correct)))
print("Number of Incorrect Classifications: {}".format(len(incorrect)))
print("Total Accuracy: {}".format((len(correct) / len(label_validation)) * 100))

np.savez('plotting.npz', res=result, redval=reduced_val, lv=label_validation)