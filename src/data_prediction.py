import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.python.keras.engine.sequential import Sequential


def tf_model(input_dims, output_dims) -> Sequential:
    model = K.Sequential([
        L.Input(shape=input_dims),
        L.Dense(200, activation='relu'),
        L.Dense(output_dims, activation='relu')
    ])
    model.compile(optimizer=K.optimizers.Adam(
        learning_rate=0.01), metrics=['accuracy', 'mae'])
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
