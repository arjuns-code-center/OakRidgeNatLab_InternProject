import numpy as np
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, UpSampling2D, Flatten, Dense
from tensorflow import keras
import horovod.tensorflow.keras as hvd

npzfile = np.load('/gpfs/alpine/gen150/scratch/arjun2612/ORNL_Coding/Code/cae/savefile.npz')
trainset = npzfile['train']
valset = npzfile['val']
lt_onehot = npzfile['ltoh']
label_validation = npzfile['labval']

hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

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
    
opt = tf.keras.optimizers.Adam(0.001 * hvd.size())
opt = hvd.DistributedOptimizer(opt)
classification_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

print(str(time.ctime()) + ": Successfully created Classification Model")
        
train_X, valid_X, train_label, valid_label = train_test_split(trainset, lt_onehot, test_size=0.2, random_state=0)
# train_X = samples x 24 x 24 x 1
# valid_X = samples x 24 x 24 x 1
# train_label = samples x 3
# valid_label = samples x 3

print(str(time.ctime()) + ": Training Classification Model...")

epochs = 20
batch_size = 64
# early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=5, restore_best_weights=True)
callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

# if hvd.rank() == 0:
#     callbacks.append(tf.keras.callbacks.ModelCheckpoint('./best_checkpoint-{epoch}.h5', monitor='val_categorical_accuracy', mode='max', save_best_only=True))

classify_labels = classification_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=callbacks, validation_data=(valid_X, valid_label))

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

np.savez('plotting.npz', res=predicted, labval=label_validation)