
import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.python.keras.engine.sequential import Sequential
import os

from tensorflow.python.keras.layers.core import Dropout
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def tf_model(input_dims, output_dims, renorm_left=None, renorm_right=None) -> Sequential:
    model = K.Sequential([
        L.Input(shape=input_dims),
        L.LSTM(10, return_sequences=True),
        L.Dropout(0.1),
        L.LSTM(10, return_sequences=True),
        L.BatchNormalization(),
        L.Dense(10, activation='relu'),
    ])
    if output_dims == 1:
        model.add(L.Dense(output_dims, activation='tanh'))
        model.compile(optimizer=K.optimizers.Adam(
            learning_rate=0.001), loss='mse', metrics=['rmse', 'mae'])
    else:
        model.add(L.Dense(output_dims, activation='softmax'))
        model.compile(optimizer=K.optimizers.Adam(
            learning_rate=0.0001), loss=K.losses.CategoricalCrossentropy(), metrics=['accuracy', 'mae'])

    print(model.summary())
    return model


def sklearn_knn():
    pass


def logistic_regression():
    pass


def linear_regression():
    pass


def polynomial_regression():
    pass


def sklearn_svm():
    pass
