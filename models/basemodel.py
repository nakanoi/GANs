import os, pickle
import numpy as np
import imageio
from glob import glob

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
            optimizer = RMSprop(learning_rate=lr)
        elif opt == 'sgd':
            optimizer = SGD(learning_rate=lr)
        elif opt == 'adagrad':
            optimizer = Adagrad(learning_rate=lr)
        elif opt == 'adadelta':
            optimizer = Adadelta(learning_rate=lr)
        elif opt == 'adam' and self.beta1:
            optimizer = Adam(learning_rate=lr, beta_1=self.beta1)
        else:
            optimizer = Adam(learning_rate=lr)

        return optimizer

    def activation(self, act, alpha=0.2):
        act = act.lower()
        if act == 'leakyrelu':
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

    def save_weights(self, folder='.', file_name='weights.h5'):
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
            file_name = os.path.join(folder, 'models', name + '.h5')
            model.load_weights(file_name)


class DataLoader:
    def __init__(self, dataset, shape=(256, 256)):
        self.dataset = dataset
        self.shape = shape

        section = 'gan'
        self.folder = './saved/{}/'.format(section) + '_' + dataset
        os.makedirs(os.path.join(self.folder, 'graph'), exist_ok=True)
        os.makedirs(os.path.join(self.folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.folder, 'weights'), exist_ok=True)

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = 'train' if not is_testing else 'val'
        path_A = glob('./datasets/{}/{}A/*'.format(self.dataset, data_type))
        path_B = glob('./datasets/{}/{}B/*'.format(self.dataset, data_type))
        self.n_batches = min(len(path_A), len(path_B)) // batch_size

        path_A = np.random.choice(path_A,
                                  self.n_batches * batch_size,
                                  replace=False)
        path_B = np.random.choice(path_B, 
                                  self.n_batches * batch_size,
                                  replace=False)

        for i in range(self.n_batches - 1):
            batch_A = path_A[i * batch_size:(i + 1) * batch_size]
            batch_B = path_B[i * batch_size:(i + 1) * batch_size]
            imgs_A = np.empty((batch_size, self.shape[0], self.shape[1], 3))
            imgs_B = np.empty((batch_size, self.shape[0], self.shape[1], 3))

            for i, (img_A, img_B) in enumerate(zip(batch_A, batch_B)):
                img_A = imageio.imread(img_A, pilmode='RGB').astype(np.uint8)
                img_B = imageio.imread(img_B, pilmode='RGB').astype(np.uint8)
                imgs_A[i] = np.array(img_A) / 127.5 - 1.0
                imgs_B[i] = np.array(img_B) / 127.5 - 1.0

                if not is_testing and np.random.random() > 0.5:
                    imgs_A[i] = np.fliplr(imgs_A[i])
                    imgs_B[i] = np.fliplr(imgs_B[i])

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = imageio.imread(path, pilmode='gray').astype(np.uint8)
        img = np.array(img) / 127.5 - 1.0

        return img[np.newaxis, :, :, :]

    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = 'train{}'.format(domain) if not is_testing else 'test{}'.format(domain)
        path = glob('./datasets/{}/{}/*'.foramt(self.dataset, data_type))
        batch_images = np.random.choice(path, size=batch_size)
        imgs = np.array([0 for _ in range(batch_size)])

        for i, path in enumerate(batch_images):
            img = imageio.imread(path, pilmode='RGB').astype(np.uint8)

            if is_testing:
                imgs[i] = np.array(img) / 127.5 - 1.0
            else:
                imgs[i] = np.array(img) / 127.5 - 1.0
                imgs[i] = np.fliplr(imgs[i]) if np.random.random() > 0.5 else imgs[i]

        return imgs

