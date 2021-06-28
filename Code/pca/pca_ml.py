import numpy as np
import time, argparse
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

args = argparse.ArgumentParser()
args.add_argument('--dataset', default='SARSMERSCOV2', type=str, help='type of data loading in')
args = args.parse_args()
datatype = args.dataset

def create_model(output_size):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(2,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['categorical_accuracy'])
    return model

npzfile1 = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/pca/sk_clusterfiles.npz')
pcamodel = None
if datatype == 'SARSMERSCOV2':
    npzfile2 = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/sars_mers_cov2_dataset/smc2_dataset.npz')
    pcamodel = create_model(3)
elif datatype == 'HEA':
    npzfile2 = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/hea_dataset/hea_dataset.npz')
    pcamodel = create_model(5)
    
print(str(time.ctime()) + ": Implementing PCA ML...")

reduced_train = npzfile1['redtrain']
reduced_val = npzfile1['redval']
lt_onehot = npzfile2['ltoh']
lv_onehot = npzfile2['lvoh']
label_validation = npzfile2['labval']

batch_size = 64
epochs = 20
early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=5, restore_best_weights=True)

history = pcamodel.fit(reduced_train, lt_onehot, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(reduced_val, lv_onehot), callbacks=early_stop)

print(str(time.ctime()) + ": Finished PCA ML")
print(str(time.ctime()) + ": Predicting with PCA Model...")

result = pcamodel.predict(reduced_val)
result = np.argmax(np.round(result), axis=1)

print(str(time.ctime()) + ": Finished predictions!")

l = np.array([]).astype(int)
for i in range(len(lv_onehot)):
    l = np.append(l, np.argmax(lv_onehot[i]))
    
correct = np.where(result == l)[0]
incorrect = np.where(result != l)[0]
print("Number of Correct Classifications: {}".format(len(correct)))
print("Number of Incorrect Classifications: {}".format(len(incorrect)))
print("Total Accuracy: {}".format((len(correct) / len(label_validation)) * 100))

np.savez('plotting.npz', res=result, redval=reduced_val, lv=label_validation)