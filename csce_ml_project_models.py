from scikeras.wrappers import KerasClassifier
import keras
keras.utils.set_random_seed(0)
import numpy as np


class NNModel():


    @staticmethod
    def _create_model(meta):
        
        import tensorflow as tf
        tf.keras.backend.clear_session()
        keras.utils.set_random_seed(0)

        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        n_features = meta["n_features_in_"]
        n_classes = meta["n_classes_"]

        act = 'swish'
        initializer = 'he_normal'
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(n_features,)))
        model.add(keras.layers.Dropout(rate=0.05))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(units=64, activation=act, kernel_initializer=initializer))
        model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(n_classes, activation='softmax',dtype='float32'))
        model.build(input_shape=(None, n_features))
        optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        loss = keras.losses.SparseCategoricalCrossentropy()
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'], )
        return model
    

    def __init__(self):
        return


    def get_model(self):
        wrapped = KerasClassifier(
            model=self._create_model,
            batch_size=32,
            epochs=2,
            verbose=0,
            callbacks=[
                keras.callbacks.CSVLogger("training_results.csv", append=True),
                ],
        )
        return wrapped
