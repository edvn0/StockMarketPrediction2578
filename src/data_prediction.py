
import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.python.keras.engine.sequential import Sequential
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def tf_model(input_dims, output_dims, mode: str) -> Sequential:
    model = K.Sequential([
        L.Input(shape=input_dims),
        L.Dense(10, activation='relu'),
        L.Dense(10, activation='relu'),
    ])
    if mode == 'regression':
        model.add(L.Dense(output_dims, activation='tanh'))
        model.compile(optimizer=K.optimizers.Adam(
            learning_rate=0.001), loss='mse', metrics=['accuracy', 'mae'])
    elif mode == 'classification':
        model.add(L.Dense(output_dims, activation='softmax'))
        model.compile(optimizer=K.optimizers.Adam(
            learning_rate=0.0001), loss=K.losses.CategoricalCrossentropy(), metrics=['accuracy', 'mae'])
    else:
        raise ValueError('Only (regression, classification) are allowed.')

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
