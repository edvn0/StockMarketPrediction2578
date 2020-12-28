
import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.python.keras.engine.sequential import Sequential
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def tf_model(input_dims, output_dims) -> Sequential:
    model = K.Sequential([
        L.Input(shape=input_dims),
        L.Conv1D(filters=60, kernel_size=5, strides=1,
                 padding="causal", activation="relu"),
        L.LSTM(60, return_sequences=True),
        L.LSTM(60, return_sequences=True),
        L.BatchNormalization(),
        L.Dense(250, activation='relu'),
    ])
    if output_dims == 1:
        model.add(L.Dense(output_dims, activation='tanh'))
        model.add(L.Lambda(lambda x: x*500))
        model.compile(optimizer=K.optimizers.Adam(
            learning_rate=0.01), loss='mse', metrics=['mse', 'mae'])
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
