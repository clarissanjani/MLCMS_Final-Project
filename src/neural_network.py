import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly
import plotly.express as px
import plotly.graph_objects as go


class NeuralNetwork:
    x_train = None
    x_test = None
    y_train = None
    y_test = None

    model = None

    def __init__(self):
        """

        """
        self.model = tf.keras.Sequential(name='grayboxann')
        self.model.add(tf.keras.Input(shape=(3,)))
        self.model.add(tf.keras.layers.Dense(10, activation='softplus', name='hiddenlayer1'))
        self.model.add(tf.keras.layers.Dense(10, activation='softplus', name='hiddenlayer2'))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='outputlayer'))

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['Accuracy', 'Precision', 'Recall'],
                           loss_weights=None,
                           weighted_metrics=None,
                           run_eagerly=None,
                           steps_per_execution=None
                           )

        self.model.summary()

    def import_data(self, file_name):
        """
        :param file_name: name of csv file that contains the SIR data
        :return: sets the training and validation data of self
        """
        df = pd.read_csv(file_name)
        x = df[['Susceptible', 'Infected', 'Recovered']]
        y = df['Mu'].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self):
        """
        :return:
        """

        x = tf.convert_to_tensor(self.x_train)
        y = tf.convert_to_tensor(self.y_train)

        self.model.fit(x,  # input data
                       y,  # target data
                       batch_size=10,
                       # Number of samples per gradient update. If unspecified, batch_size will default to 32.
                       epochs=2,
                       # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x
                       # and y data provided
                       verbose='auto',
                       # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar,
                       # 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with
                       # ParameterServerStrategy.
                       callbacks=None,
                       # default=None, list of callbacks to apply during training. See tf.keras.callbacks

                       # default=0.0, Fraction of the training data to be used as validation data. The model will set
                       # apart this fraction of the training data, will not train on it, and will evaluate the loss
                       # and any model metrics on this data at the end of each epoch.
                       shuffle=True,
                       # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for
                       # 'batch').
                       class_weight=None,
                       # default=None, Optional dictionary mapping class indices (integers) to a weight (float)
                       # value, used for weighting the loss function (during training only). This can be useful to
                       # tell the model to "pay more attention" to samples from an under-represented class.
                       sample_weight=None,
                       # default=None, Optional Numpy array of weights for the training samples, used for weighting
                       # the loss function (during training only).
                       initial_epoch=0,
                       # Integer, default=0, Epoch at which to start training (useful for resuming a previous
                       # training run).
                       steps_per_epoch=10,
                       # Integer or None, default=None, Total number of steps (batches of samples) before declaring
                       # one epoch finished and starting the next epoch. When training with input tensors such as
                       # TensorFlow data tensors, the default None is equal to the number of samples in your dataset
                       # divided by the batch size, or 1 if that cannot be determined.
                       validation_steps=None,
                       # Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps
                       # (batches of samples) to draw before stopping when performing validation at the end of every
                       # epoch.
                       validation_batch_size=None,
                       # Integer or None, default=None, Number of samples per validation batch. If unspecified,
                       # will default to batch_size.
                       validation_freq=3,
                       # default=1, Only relevant if validation data is provided. If an integer, specifies how many
                       # training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs
                       # validation every 2 epochs.
                       max_queue_size=10,
                       # default=10, Used for generator or keras.utils.Sequence input only. Maximum size for the
                       # generator queue. If unspecified, max_queue_size will default to 10.
                       workers=1,
                       # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of
                       # processes to spin up when using process-based threading. If unspecified, workers will
                       # default to 1.
                       use_multiprocessing=False
                       # default=False, Used for generator or keras.utils.Sequence input only. If True,
                       # use process-based threading. If unspecified, use_multiprocessing will default to False.
                       )

    def predict(self):
        """
        :return:
        """
        x1 = self.model.predict(self.s_train)
        x2 = self.model.predict(self.s_test)
        y1 = self.model.predict(self.i_train)
        y2 = self.model.predict(self.i_test)

        return x1, x2, y1, y2
