import os, pickle
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import (Adam, RMSprop, SGD, Adagrad,
                                         Adadelta)
from tensorflow.keras.layers import LeakyReLU, ReLU, Activation
from tensorflow.keras.utils import plot_model


class BaseModel:
    def __init__(self):
        self.models = {}

    def optimizer(self, lr):
        opt = self.opt.lower()
        if opt == 'rmsprop':
            optimizer = RMSprop(lr=lr)
        elif opt == 'sgd':
            optimizer = SGD(lr=lr)
        elif opt == 'adagrad':
            optimizer = Adagrad(lr=lr)
        elif opt == 'adadelta':
            optimizer = Adadelta(lr=lr)
        elif opt == 'adam' and self.beta1:
            optimizer = Adam(lr=lr, beta_1=self.beta1)
        else:
            optimizer = Adam(lr=lr)

        return optimizer

    def activation(self, act, alpha=0.2):
        act = act.lower()
        if act == 'leakrelu':
            activation = LeakyReLU(alpha=alpha)
        elif act == 'relu':
            activation = ReLU()
        else:
            activation = Activation(act)

        return activation

    def trainable(self, model, tf):
        model.trainable = tf
        for layer in model.layers:
            layer.trainable = tf

    def summary(self):
        for name, model in self.models.items():
            print('############### %s ###############' % (name))
            model.summary()

    def plot_models(self, folder='.'):
        folder = os.path.join(folder, 'plots')
        os.makedirs(folder, exist_ok=True)

        for name, model in self.models.items():
            file_name = os.path.join(folder, name + '.png')
            plot_model(model, to_file=file_name, show_shapes=True)

    def save_params(self, folder='.'):
        os.makedirs(folder, exist_ok=True)

        file_name = os.path.join(folder, 'params.pkl')
        with open(file_name, 'wb') as f:
            pickle.dumps(*self.params, f)

    def save_weights(self, folder='.', file_name='weights.ht'):
        folder = os.path.join(folder, 'weights')
        os.makedirs(folder, exist_ok=True)

        for name, model in self.models.items():
            file = os.path.join(folder + name + file_name)
            model.save_weights(file)

    def save_models(self, folder='.'):
        folder = os.path.join(folder, 'models')
        os.makedirs(folder, exist_ok=True)

        for name, model in self.models.items():
            file_name = os.path.join(folder, name + '.png')
            model.save(file_name)

    def load_models(self, folder='.'):
        for name, model in self.models.imtes():
            file_name = os.path.join(folder, 'models', name + '.png')
            model.load_weights(file_name)

