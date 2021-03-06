import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import imageio
from glob import glob

import tensorflow_datasets as tfds
from tensorflow.keras.optimizers import (Adam, RMSprop, SGD, Adagrad,
                                         Adadelta)
from tensorflow.keras.layers import LeakyReLU, ReLU, Activation
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist


class BaseModel:
    def __init__(self):
        self.models = {}
        self.di_real_lss = []
        self.di_fake_lss = []
        self.di_lss = []
        self.di_acc = []
        self.ge_lss = []

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

    def show_img(self, generator, z_dim, file_name, color='RGB', show=False):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, z_dim))
        gen_imgs = generator.predict(noise)
        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c, figsize=(15, 15))
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(np.squeeze(gen_imgs[cnt, :, :, :]), cmap=color)
                axs[i, j].axis('off')
                cnt += 1
        if show:
            plt.show()
        fig.savefig(os.path.join(self.loader.folder, 'images', file_name))
        plt.close()

    def plot_models(self):
        folder = os.path.join(self.loader.folder, 'plots')
        os.makedirs(folder, exist_ok=True)

        for name, model in self.models.items():
            file_name = os.path.join(folder, name + '.png')
            plot_model(model, to_file=file_name, show_shapes=True)

    def save_params(self):
        folder = os.path.join(self.loader.folder, 'params')
        os.makedirs(folder, exist_ok=True)

        file_name = os.path.join(folder, 'params.pkl')
        with open(file_name, 'wb') as f:
            pickle.dumps(*self.params, f)

    def save_weights(self, file_name='weights.h5'):
        folder = os.path.join(self.loader.folder, 'weights')
        os.makedirs(folder, exist_ok=True)

        for name, model in self.models.items():
            file = os.path.join(folder, name + file_name)
            model.save_weights(file)

    def save_models(self, epoch=''):
        folder = os.path.join(self.loader.folder, 'models')
        os.makedirs(folder, exist_ok=True)

        for name, model in self.models.items():
            file_name = os.path.join(folder, 'models', '{}{}.h5'.format(name, epoch))
            model.save(file_name)

    def load_models(self, epoch=''):
        folder = os.path.join(self.loader.folder, 'models')
        for name, model in self.models.items():
            file_name = os.path.join(folder, 'models', '{}{}.h5'.format(name, epoch))
            model.load_weights(file_name)

    def plot_loss(self):
        fig = plt.figure(figsize=(150, 100))

        ax1 = fig.add_subplot(231)
        ax1.set_xlim([0, len(self.di_real_lss)])
        ax1.set_title('Discriminator Real Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.plot(len(self.di_real_lss), self.di_real_lss)

        ax2 = fig.add_subplot(232)
        ax2.set_xlim([0, len(self.di_fake_lss)])
        ax2.set_title('Discriminator Fake Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.plot(len(self.di_fake_lss), self.di_fake_lss)

        ax3 = fig.add_subplot(233)
        ax3.set_xlim([0, len(self.di_lss)])
        ax3.set_title('Discriminator Loss')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Loss')
        ax3.plot(len(self.di_lss), self.di_lss)

        ax4 = fig.add_subplot(234)
        ax4.set_xlim([0, len(self.ge_lss)])
        ax4.set_title('Generator Loss')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Loss')
        ax4.plot(len(self.ge_lss), self.ge_lss)

        ax5 = fig.add_subplot(235)
        ax5.set_xlim([0, len(self.di_acc)])
        ax5.set_ylim([0, 100])
        ax5.set_title('Discriminator Accuracy')
        ax5.set_xlabel('Epochs')
        ax5.set_ylabel('Accuracy')
        ax5.plot(len(self.di_acc), self.di_acc)

        plt.show()
        plt.cla()
        plt.clf()


class DataLoader:
    def __init__(self, dataset, ID, shape=(256, 256), color='RGB', section='GAN'):
        self.dataset = dataset
        self.shape = shape
        self.color = color

        self.folder = './run/{}/{}_{}'.format(section, ID, dataset)
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
                img_A = imageio.imread(img_A, pilmode=self.color).astype(np.uint8)
                img_B = imageio.imread(img_B, pilmode=self.color).astype(np.uint8)
                imgs_A[i] = np.array(img_A) / 127.5 - 1.0
                imgs_B[i] = np.array(img_B) / 127.5 - 1.0

                if not is_testing and np.random.random() > 0.5:
                    imgs_A[i] = np.fliplr(imgs_A[i])
                    imgs_B[i] = np.fliplr(imgs_B[i])

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = imageio.imread(path, pilmode=self.color).astype(np.uint8)
        img = np.array(img) / 127.5 - 1.0

        return img[np.newaxis, :, :, :]

    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = 'train{}'.format(domain) if not is_testing else 'test{}'.format(domain)
        path = glob('./datasets/{}/{}/*'.format(self.dataset, data_type))
        batch_images = np.random.choice(path, size=batch_size)
        imgs = []

        for i, path in enumerate(batch_images):
            img = imageio.imread(path, pilmode=self.color).astype(np.uint8)

            if is_testing:
                imgs.append(np.array(img) / 127.5 - 1.0)
            else:
                imgs.append(np.array(img) / 127.5 - 1.0)
                imgs[-1] = np.fliplr(imgs[-1]) if np.random.random() > 0.5 else imgs[-1]

        return np.array(imgs)

    def load_np_data(self, data_type):
        mypath = os.path.join("./datasets", data_type)
        txt_name_list = []
        for (dirpath, dirnames, filenames) in os.walk(mypath):
            for f in filenames:
                txt_name_list.append(f)
                break

        slice_train = int(80000/len(txt_name_list))
        i = 0
        x_total, y_total = None, None

        for txt_name in txt_name_list:
            txt_path = os.path.join(mypath,txt_name)
            x = np.load(txt_path)
            x = (x.astype('float32') - 127.5) / 127.5
            x = x.reshape(x.shape[0], 28, 28, 1)
            y = [i] * len(x)  
            np.random.shuffle(x)
            np.random.shuffle(y)
            x = x[:slice_train]
            y = y[:slice_train]

            if i != 0: 
                x_total = np.concatenate((x, x_total), axis=0)
                y_total = np.concatenate((y, y_total), axis=0)
            else:
                x_total = x
                y_total = y

            i += 1

        return x_total, y_total

    def load_mnist(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = (x_train.astype('float32') - 127.5) / 127.5
        x_train = x_train.reshape(x_train.shape + (1, ))
        x_test = (x_test.astype('float32') - 127.5) / 127.5
        x_test = x_test.reshape(x_test.shape + (1, ))

        return (x_train, y_train), (x_test, y_test)

    def load_tf_data(self, split=None, nd=False):
        ds = tfds.load(self.dataset, ) if split is None else tfds.load(self.dataset, split=split)
        ds = tfds.as_numpy(ds) if nd else ds

        return ds
