import os
import numpy as np
import tensorflow as tf
import re
from tensorflow import keras
import datetime
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Dropout, Input, Flatten, Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
from sklearn import metrics
from One_dimensional_Local_Gradient_Pattern import ODLGP
from Local_Neighbor_Descriptive_Pattern import LNDP
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from preprocess import clean_signal
path = "G:\Facultate\licenta\Code_and_dataset\dataset"
folders = []
x_train_lndp, x_val_lndp, x_test_lndp, y_train_lndp, y_val_lndp, y_test_lndp = [], [], [], [], [], []
x_train_lgp, x_val_lgp, x_test_lgp, y_train_lgp, y_val_lgp, y_test_lgp = [], [], [], [], [], []

window_size=4096
neighbours = 8
fs = 173.61  # Sample frequency (Hz)
f0 = 50  # Frequency to be removed from signal (Hz)
Q = 50  # Quality factor
stride =0
#----------------load data from folders---------
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
        #bring the data values in [0,1]
        num_data= clean_signal(num_data,fs,f0,Q)
        num_data= (num_data-min(num_data))/(max(num_data)-min(num_data))
        #print(num_data)
        #print(num_data.shape)
        for i in range(0, len(num_data)-window_size, window_size):
            window = num_data[i:i+window_size]
            lgp = np.reshape(ODLGP(window, neighbours,stride), (np.power(2,neighbours), 1))
            lgp=np.transpose(lgp)

            lndp = np.reshape(LNDP(window, neighbours,stride), (np.power(2,neighbours), 1))
            lndp=np.transpose(lndp)

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
                x_train_lndp.append(lndp)
                y_train_lndp.append(label)
                x_train_lgp.append(lgp)
                y_train_lgp.append(label)
            elif folder == 'val_data':
                x_val_lndp.append(lndp)
                y_val_lndp.append(label)
                x_val_lgp.append(lgp)
                y_val_lgp.append(label)
            elif folder == 'test_data':
                x_test_lndp.append(lndp)
                y_test_lndp.append(label)
                x_test_lgp.append(lgp)
                y_test_lgp.append(label)

#------------------convert to binary values---------------
y_train_lndp = tf.keras.utils.to_categorical(y_train_lndp, 3)
y_test_lndp = tf.keras.utils.to_categorical(y_test_lndp,3 )
y_val_lndp = tf.keras.utils.to_categorical(y_val_lndp, 3)
y_train_lgp = tf.keras.utils.to_categorical(y_train_lgp, 3)
y_test_lgp = tf.keras.utils.to_categorical(y_test_lgp,3 )
y_val_lgp = tf.keras.utils.to_categorical(y_val_lgp, 3)

scaler=StandardScaler()
x_train_lndp=np.asanyarray(x_train_lndp)
x_train_lndp= np.reshape(x_train_lndp,(x_train_lndp.shape[0],x_train_lndp.shape[2] ))
x_train_lndp = scaler.fit_transform(np.asanyarray(x_train_lndp))

x_train_lgp=np.asanyarray(x_train_lgp)
x_train_lgp= np.reshape(x_train_lgp,(x_train_lgp.shape[0],x_train_lgp.shape[2] ))
x_train_lgp = scaler.fit_transform(np.asanyarray(x_train_lgp))

x_val_lndp=np.asanyarray(x_val_lndp)
x_val_lndp= np.reshape(x_val_lndp,(x_val_lndp.shape[0],x_val_lndp.shape[2] ))
x_val_lndp = scaler.fit_transform(np.asanyarray(x_val_lndp))

x_val_lgp=np.asanyarray(x_val_lgp)
x_val_lgp= np.reshape(x_val_lgp,(x_val_lgp.shape[0],x_val_lgp.shape[2] ))
x_val_lgp = scaler.fit_transform(np.asanyarray(x_val_lgp))

x_test_lndp=np.asanyarray(x_test_lndp)
x_test_lndp= np.reshape(x_test_lndp,(x_test_lndp.shape[0],x_test_lndp.shape[2] ))
x_test_lndp = scaler.fit_transform(np.asanyarray(x_test_lndp))


x_test_lgp=np.asanyarray(x_test_lgp)
x_test_lgp= np.reshape(x_test_lgp,(x_test_lgp.shape[0],x_test_lgp.shape[2] ))
x_test_lgp = scaler.fit_transform(np.asanyarray(x_test_lgp))

def DNN():
    #two sets of inputs -----------------------------------
    input_lndp = Input(shape=(1,256), name='input_lndp')
    input_lgp = Input(shape=(1,256), name='input_lgp')

    #lndp branch --------------------------------------
    hidden = Dense(256, activation='relu', name='relu1', kernel_regularizer=regularizers.l1_l2(0.01, 0.01))(input_lndp)
    out_lndp = Dropout(0.3, name='drop1')(hidden)
    lndp_model = tf.keras.Model(inputs=input_lndp, outputs=out_lndp)

    #lgp branch --------------------------------------------
    hidden = Dense(256, activation='relu', name='relu2', kernel_regularizer=regularizers.l1_l2(0.01, 0.01))(input_lgp)
    out_lgp = Dropout(0.3, name='drop2')(hidden)
    #out_lgp = Dense(64, activation='relu', name='relu4', kernel_regularizer=regularizers.l1_l2(0.1, 0.1))(out_lgp)

    lgp_model = tf.keras.Model(inputs=input_lgp, outputs=out_lgp)

    #concat --------------------------------------------------
    combine = Concatenate()([lndp_model.output, lgp_model.output])
    output = Dense(3, activation='softmax', name='fc2')(combine)
    model = tf.keras.Model(inputs=[input_lndp,input_lgp], outputs=output)

    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,
        decay_rate=0.96,
        staircase=False)
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

checkpoint_filepath = 'G:\Facultate\licenta\Code_and_dataset\\NEW\lndp_lgp_concat\\noscaler\\4096\\0%\checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1)

log_dir = f"G:\Facultate\licenta\Code_and_dataset\\NEW\lndp_lgp_concat\\noscaler\\4096\\0%\log"\
          + datetime.datetime.now().strftime(f"%Y%m%d-%H%M%S-3out_256_03drop_lrdecay_neighbours{neighbours}_stride{stride}")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

model = DNN()
# #model = load_model("model256128_regneurons_lrdecay_01_drop03_4096_neighbours8_98test.h5")
model.fit(x=[x_train_lndp, x_train_lgp], y=y_train_lndp,epochs=1000, verbose=2, batch_size=256,
          validation_data=([x_val_lndp, x_val_lgp], y_val_lndp),
          callbacks=[model_checkpoint_callback, tensorboard_callback ,tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=64,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)])
model.save(f"G:\Facultate\licenta\Code_and_dataset\\NEW\lndp_lgp_concat\\noscaler\\4096\\0%\model3out_256_03drop_lrdecay_neighbours{neighbours}_stride{stride}.h5")

y_predict =model.evaluate(x=[x_test_lndp,x_test_lgp],y = y_test_lndp ,verbose=1)

