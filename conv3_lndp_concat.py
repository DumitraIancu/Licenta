import os
import numpy as np
import tensorflow as tf
import re
from tensorflow import keras
import datetime
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Dropout, Input, Flatten, Activation, Concatenate
from tensorflow.keras import regularizers
from preprocess import clean_signal
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from One_dimensional_Local_Gradient_Pattern import ODLGP

path = "G:\Facultate\licenta\Code_and_dataset\dataset"
folders = []
x_train, x_val, x_test, y_train, y_val, y_test = [], [], [], [], [], []
x_train_lgp, x_val_lgp, x_test_lgp, y_train_lgp, y_val_lgp, y_test_lgp = [], [], [], [], [], []
neighbours = 8
window_size=256
fs = 173.61  # Sample frequency (Hz)
f0 = 50  # Frequency to be removed from signal (Hz)
Q = 50  # Quality factor


#----------------load data from folders----------------------------------

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
        num_data_clean= clean_signal(num_data,fs,f0,Q)
        num_data_clean= (num_data_clean-min(num_data_clean))/(max(num_data_clean)-min(num_data_clean))
        for i in range(0, len(num_data_clean)-window_size, window_size//2):
            window = num_data_clean[i:i+window_size]
            lgp = np.reshape(ODLGP(window, neighbours, 2), (np.power(2, neighbours), 1))
            lgp = np.transpose(lgp)
            #print(window.shape)

            # loading the label - y
            if re.search("^Z|^O", file):
                label = 0
                # healthy
            elif re.search("^S", file):
                label = 1  # seizure
            else:
                label = 2  # non- seizure

            # creating the lists
            if folder == 'train_data':
                x_train_lgp.append(lgp)
                y_train_lgp.append(label)
                x_train.append(window)
                y_train.append(label)
            elif folder == 'val_data':
                x_val_lgp.append(lgp)
                y_val_lgp.append(label)
                x_val.append(window)
                y_val.append(label)
            elif folder == 'test_data':
                x_test_lgp.append(lgp)
                y_test_lgp.append(label)
                x_test.append(window)
                y_test.append(label)

#--convert to binary values----
y_train = tf.keras.utils.to_categorical(y_train,3)
y_test = tf.keras.utils.to_categorical(y_test,3)
y_val = tf.keras.utils.to_categorical(y_val, 3)

y_train_lgp = tf.keras.utils.to_categorical(y_train_lgp,3)
y_test_lgp = tf.keras.utils.to_categorical(y_test_lgp,3)
y_val_lgp = tf.keras.utils.to_categorical(y_val_lgp, 3)

#normalize---------------------------------------------------------------
scaler=MinMaxScaler()

x_train_lgp=np.asanyarray(x_train_lgp)
x_train_lgp= np.reshape(x_train_lgp,(x_train_lgp.shape[0],x_train_lgp.shape[2] ))
x_train_lgp = scaler.fit_transform(np.asanyarray(x_train_lgp))

x_val_lgp=np.asanyarray(x_val_lgp)
x_val_lgp= np.reshape(x_val_lgp,(x_val_lgp.shape[0],x_val_lgp.shape[2] ))
x_val_lgp = scaler.fit_transform(np.asanyarray(x_val_lgp))

x_test_lgp=np.asanyarray(x_test_lgp)
x_test_lgp= np.reshape(x_test_lgp,(x_test_lgp.shape[0],x_test_lgp.shape[2] ))
x_test_lgp = scaler.fit_transform(np.asanyarray(x_test_lgp))

x_train=np.asanyarray(x_train)
y_train=np.asanyarray(y_train)
x_val=np.asanyarray(x_val)
y_val=np.asanyarray(y_val)
x_test =np.asanyarray(x_test)
y_test=np.asanyarray(y_test)
print(x_val.shape, x_val_lgp.shape, y_val.shape)
print(x_train.shape, x_train_lgp.shape, y_train.shape)

def concat_model_train():

    input = Input(shape=(window_size, 1) , name='input')
    input_lgp = Input(shape=(256), name='input_lgp')

#-------------convolutional branch--------------------------------------
    hidden = Conv1D(filters=24, kernel_size=5,strides=4, name='conv1', kernel_regularizer=regularizers.l1_l2(0.01,0.001)) (input)
    hidden = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(hidden)
   # hidden = Dense(24, activation='relu', name='relu1', kernel_regularizer=regularizers.l1_l2(0.01,0.01))(hidden)
    hidden = Activation('relu')(hidden)

    hidden = Conv1D(filters=16, kernel_size=5, strides=3, name='conv2', kernel_regularizer=regularizers.l1_l2(0.01,0.001))(hidden)
    hidden = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(hidden)
    hidden = Activation('relu')(hidden)

    #dropout = Dropout(0.5, name='drop1')(hidden)

    hidden = Conv1D(filters=8, kernel_size=3, strides=2, name='conv3', kernel_regularizer=regularizers.l1_l2(0.01,0.001) )(hidden)
    hidden = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(hidden)
    hidden = Activation('relu')(hidden)

    hidden = Flatten()(hidden)
    hidden = Dense(64, activation='relu', name='fc2', kernel_regularizer=regularizers.l1_l2(0.1,0.1))(hidden)
    hidden = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(hidden)
    dropout = Dropout(0.5, name='drop1')(hidden)
    conv_model = tf.keras.Model(inputs=input, outputs=dropout)

    #----------------------------------- lgp branch --------------------------------------
    hidden = Dense(64, activation='relu', name='relu1', kernel_regularizer=regularizers.l1_l2(0.1, 0.1))(input_lgp)
    hidden = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(hidden)
    out_lgp = Dropout(0.5, name='drop2')(hidden)
    lgp_model = tf.keras.Model(inputs=input_lgp, outputs=out_lgp)


    combine = Concatenate()([conv_model.output, lgp_model.output ])
    output = Dense(3, activation='softmax', name='fc3')(combine)


    model = tf.keras.Model(inputs=[input, input_lgp], outputs=output)

    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,
        decay_rate=0.9,
        staircase=False)
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


checkpoint_filepath = f'G:\Facultate\licenta\Code_and_dataset\\NEW\conv3_concatlgp\\{window_size}_{neighbours}\minmaxscaler\\50%\checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1)

log_dir = f"G:\Facultate\licenta\Code_and_dataset\\NEW\conv3_concatlgp\\{window_size}_{neighbours}\minmaxscaler\\50%\log" \
          + datetime.datetime.now().strftime("%Y%m%d-%H%M%S_64neurons_lrdecay_reg0001_lgp")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
early_stoppping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=64,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False)
model = concat_model_train()
#model.fit(x = [x_train, x_train_lgp], y= y_train,epochs=1000, verbose=2, batch_size=128,
          #validation_data=([x_val, x_val_lgp], y_val),
          #callbacks=[model_checkpoint_callback, tensorboard_callback, early_stoppping_callback ])

#model.save(f"G:\Facultate\licenta\Code_and_dataset\\NEW\conv3_concatlgp\\{window_size}_{neighbours}\minmaxscaler\\50%\model164neurons_lrdecay_reg001_0001_lgp.h5")
y_predict =model.evaluate([x_test, x_test_lgp], y_test, verbose=1)
print()
y_predict_0= np.argmax(model.predict([x_test, x_test_lgp]),axis=1)
x=confusion_matrix(np.argmax(y_test,axis=1), y_predict_0)
display = ConfusionMatrixDisplay(x,display_labels=["10","01"]).plot()
plt.show()


