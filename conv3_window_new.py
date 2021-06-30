import os
import numpy as np
import tensorflow as tf
import re
from tensorflow import keras
import datetime
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Dropout, Input, Flatten
from tensorflow.keras import regularizers
from preprocess import clean_signal
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

path = "G:\Facultate\licenta\Code_and_dataset\dataset"
folders = []
x_train, x_val, x_test, y_train, y_val, y_test = [], [], [], [], [], []
window_size=256
fs = 173.61  # Sample frequency (Hz)
f0 = 50  # Frequency to be removed from signal (Hz)
Q = 50  # Quality factor
#----------------load data from folders-------------------------------------------------------------------------------

directory_contents = os.listdir(path)
for item in directory_contents:
    if os.path.isdir(os.path.join(path, item)):
        folders.append(item)
for folder in folders:
    new_path = os.path.join(path, folder)
    files = os.listdir(new_path)
    for file in files:
        # loading the data - x
        num_data = np.loadtxt(os.path.join(new_path, file))
        num_data=clean_signal(num_data,fs,f0,Q)
        #bring the data values in [0,1]
        num_data= (num_data-min(num_data))/(max(num_data)-min(num_data))

        for i in range(0, len(num_data)-window_size, window_size//2):
            window = num_data[i:i+window_size]
            #print(window.shape)

            # loading the label - y
            if re.search("^Z|^O", file):
                label = 0
                # healthy
            elif re.search("^S", file):
                label = 1 # seizure
            else:
                label =2 # non- seizure

            # creating the lists
            if folder == 'train_data':
                x_train.append(window)
                y_train.append(label)
            elif folder == 'val_data':
                x_val.append(window)
                y_val.append(label)
            elif folder == 'test_data':
                x_test.append(window)
                y_test.append(label)

#--------------------------------------------------------------------------------
#--convert to binary values----
y_train = tf.keras.utils.to_categorical(y_train, 3)
y_test = tf.keras.utils.to_categorical(y_test,3 )
y_val = tf.keras.utils.to_categorical(y_val, 3)
#-------------------building the datasets----------------------------------------
train_dataset = tf.data.Dataset.from_tensor_slices((np.asanyarray(x_train), np.asanyarray(y_train)))
train_dataset = train_dataset.shuffle(5000).batch(256)
val_dataset = tf.data.Dataset.from_tensor_slices((np.asanyarray(x_val), np.asanyarray(y_val)))
val_dataset = val_dataset.shuffle(5000).batch(256)
test_dataset = tf.data.Dataset.from_tensor_slices((np.asanyarray(x_test), np.asanyarray(y_test)))
test_dataset = test_dataset.batch(256)

def cnn_model_train():

    input = Input(shape=(window_size, 1) , name='input')

    hidden = Conv1D(filters=24, kernel_size=5,strides=4, name='conv1', activation='relu' ,kernel_regularizer=regularizers.l1_l2(0.01,0.001)) (input)
    hidden = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(hidden)

    hidden = Conv1D(filters=16, kernel_size=5, strides=3, activation='relu' , name='conv2', kernel_regularizer=regularizers.l1_l2(0.01,0.001))(hidden)
    hidden = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(hidden)

    hidden = Conv1D(filters=8, kernel_size=3, strides=2, name='conv3',activation='relu' , kernel_regularizer=regularizers.l1_l2(0.01,0.001) )(hidden)
    hidden = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(hidden)
    hidden = Flatten()(hidden)

    hidden = Dense(64, activation='relu', name='fc1', kernel_regularizer=regularizers.l1_l2(0.1,0.1))(hidden)
    hidden = Dropout(0.5, name='drop0')(hidden)

    output = Dense(3, activation='softmax', name='fc2')(hidden)
   #---------------configuring the learning rate----------------
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,
        decay_rate=0.96,
        staircase=False)
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)

    model = tf.keras.Model(inputs=input, outputs=output)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


checkpoint_filepath = f'G:\Facultate\licenta\Code_and_dataset\\NEW\conv3_bun\\{window_size}\\50%\checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1)

log_dir = f"G:\Facultate\licenta\Code_and_dataset\\NEW\conv3_bun\\{window_size}\\50%" + datetime.datetime.now().strftime("\log_%Y%m%d-%H%M%S_64neurons_lr01_reg001_batch32_stride4_3_2_05")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
early_stoppping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=64,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False)
model = cnn_model_train()
#model = load_model("G:\Facultate\licenta\Code_and_dataset\checkpoints\models")
model.fit(train_dataset,epochs=2000, verbose=2, batch_size=32, validation_data=val_dataset, callbacks=[model_checkpoint_callback, tensorboard_callback, early_stoppping_callback ])
model.save(f"G:\Facultate\licenta\Code_and_dataset\\NEW\conv3_bun\\{window_size}\\50%\model64_432_batch32_reg001_lrdecay.h5")

y_predict =model.evaluate(test_dataset, verbose=1)
y_predict_0= np.argmax(model.predict(test_dataset),axis=1)
x=confusion_matrix(np.argmax(y_test,axis=1), y_predict_0)
display = ConfusionMatrixDisplay(x,display_labels=["100","010","001"]).plot()
plt.show()

