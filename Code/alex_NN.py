import pandas as pd
import numpy as np

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             LearningRateScheduler, TensorBoard)
from keras.optimizers import SGD, Adam

dat = pd.read_csv('../Data/dat_features.csv')

dat = dat.loc[:, "Access":]
print(dat.columns)

unseen = dat[dat.Impressions.eq(0.0)]
training = dat[dat.Impressions > 0.0]

labels = [
        'Q119', 'Q219',
        'Q319', 'Q419',
        'BP', 'DC', 'DE', 'DP',
        'GD', 'GX', 'PL', 'PM',
        'PN', 'PT', 'SR', 'SV',
        'TN', 'VE',
        'Length',
        'Spot_Cost',
        'Cable',
        'DirecTV',
        'Dish_Network',
        'National_Network',
        "Over-the-top_content",
        'Q1', 'Q2', 'Q3', 'Q4',
        'bin_1', 'bin_2',
        'bin_3', 'bin_4', 'bin_5',
        'midnight', 'one_am', 'two_am', 'three_am', 'four_am', 'five_am',
        'six_am', 'seven_am', 'eight_am', 'nine_am', 'ten_am', 'eleven_am',
        'noon', 'one_pm', 'two_pm', 'three_pm', 'four_pm', 'five_pm', 'six_pm',
        'seven_pm', 'eight_pm', 'nine_pm', 'ten_pm', 'eleven_pm'
    ]

X = training.loc[:, labels]
y = training.loc[:, 'absolute_audience_size']

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_x.fit(X)
X_scale = scaler_x.transform(X)

scaler_y.fit(y.values.reshape(1, -1))
y_scale = scaler_y.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_scale, y_scale, test_size=0.30)

dim = len(labels)

filepath = '../nn_files/weights.hdf5'
logspath = '../nn_files/logs/scalars/' + datetime.now().strftime("%Y%m%d-%H%M%S")

def scheduler(epoch):
        if epoch < 10:
            return 0.001
        else:
            return float(0.001 * tf.math.exp(0.1 * (10 - epoch)))


callbacks = [# TensorBoard(log_dir=logspath),
             EarlyStopping(monitor='val_mean_squared_error', patience=8, mode='min'),
             ModelCheckpoint(filepath, monitor='val_mean_squared_error', verbose=1, mode='min', period=1,
                             save_best_only=True),
             ReduceLROnPlateau(monitor='val_mean_squared_error', patience=5, verbose=1, factor=0.25,
                               min_lr=0.00000001, mode='min')]

def create_model():
    model = Sequential()
    model.add(Dense(256, input_dim=dim, activation='relu'))
    model.add(Dense(256, activation='relu'))

    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))

    model.add(Dense(16, activation='relu'))

    model.add(Dense(1))
    return model

model = create_model()
print(model.summary())

ad = Adam(lr=0.001,
          amsgrad=True)
model.compile(loss='mean_squared_error',
              optimizer=ad,
              metrics=['mean_squared_error'])

model.fit(X_train, y_train,
          epochs=100, validation_split=0.3, callbacks=callbacks)
# preds_train = model.predict(X_train)
# preds_test = model.predict(X_test)
preds_train = scaler_y.inverse_transform(model.predict(X_train))
preds_test = scaler_y.inverse_transform(model.predict(X_test))

print(np.mean(preds_test))
# print('Training error', np.sqrt(MSE(scaler_y.inverse_transform(y_train), preds_train)))
# print('Test error', np.sqrt(MSE(scaler_y.inverse_transform(y_test), preds)))
print('Training error', np.sqrt(MSE(scaler_y.inverse_transform(y_train), preds_train)))
print('Test error', np.sqrt(MSE(scaler_y.inverse_transform(y_test), preds_test)))
