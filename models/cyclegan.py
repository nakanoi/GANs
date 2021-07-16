from collections import deque
import random
import matplotlib.pyplot as plt

from tensorflow import pad
from tensorflow.keras.layers import (Conv2D, Input, Layer, InputSpec,
                                     add, Conv2DTranspose, 
                                     )
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model

from models.basemodel import BaseModel, DataLoader, np, os


class ReflectionPadding2d(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] + 2 * self.padding[0],
                input_shape[2] + 2 * self.padding[1],
                input_shape[3])

    def call(self, inputs, mask=None, **kwargs):
        w_pad, h_pad = self.padding
        return pad(inputs,
                   [[0,0],
                    [h_pad,h_pad],
                    [w_pad,w_pad],
                    [0,0],
                    ],
                   'REFLECT')


class CycleGAN(BaseModel):
    def __init__(self,
                 input_dim = (28, 28, 1),
                 lr = 0.0002,
                 residual = 9,
                 buf_maxlen = 50,
                 lambda_valid = 1,
                 lambda_reconst = 10,
                 lambda_id = 2,
                 ge_filters = 64,
                 di_filters = 64,
                 opt = 'adam',
                 beta1 = 0.5,
                 start_epoch = 0,
                 k = 1,
                 data_name = '',
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.residual = residual
        self.lr = lr
        self.buf_maxlen = buf_maxlen
        self.lambda_valid = lambda_valid
        self.lambda_reconst = lambda_reconst
        self.lambda_id = lambda_id
        self.ge_filters = ge_filters
        self.di_filters = di_filters
        self.opt = opt
        self.beta1 = beta1
        self.epochs = start_epoch
        self.k = k
        self.data_name = data_name

        self.loader = DataLoader(self.data_name, tuple(input_dim[:2]))

        self.img_row = input_dim[0]
        self.img_col = input_dim[1]
        self.img_channel = input_dim[2]
        self.img_shape = tuple(input_dim[:3])

        patch = int(self.img_row / 2 ** 3)
        self.disc_patch = (patch, patch, 1)

        self.weight_init = RandomNormal(mean=0.0, stddev=0.02,)
        self.di_lss, self.ge_lss = [], []
        self.buf_A = deque(maxlen=self.buf_maxlen)
        self.buf_B = deque(maxlen=self.buf_maxlen)

        self.compile_model()

    def build_generator(self):
        def conv7s1(layer, filters, final):
            y = Conv2D(
                filters=filters,
                kernel_size=7,
                strides=1,
                padding='same',
                kernel_initializer=self.weight_init,
                )(layer)
            if final:
                y = self.activation('tanh')(y)
            else:
                y = InstanceNormalization(
                    axis=-1,
                    center=False,
                    scale=False,
                    )(y)
                y = self.activation('relu')(y)

            return y

        def downsample(layer, filters):
            y = Conv2D(
                filters=filters,
                kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer=self.weight_init,
                )(layer)
            y = InstanceNormalization(
                axis=-1,
                center=False,
                scale=False,
                )(y)
            y = self.activation('relu')(y)

            return y

        def upsample(layer, filters):
            y = Conv2DTranspose(
                filters=filters,
                kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer=self.weight_init,
                )(layer)
            y = InstanceNormalization(
                axis=-1,
                center=False,
                scale=False,
                )(y)
            y = self.activation('relu')(y)

            return y

        def residual(layer, filters):
            skip = layer
            y = ReflectionPadding2d()(layer)
            y = Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                strides=1,
                padding='valid',
                kernel_initializer=self.weight_init
                )(y)
            y = InstanceNormalization(
                axis=-1,
                center=False,
                scale=False,
                )(y)
            y = self.activation('relu')(y)

            y = ReflectionPadding2d()(layer)
            y = Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                strides=1,
                padding='valid',
                kernel_initializer=self.weight_init
                )(y)
            y = InstanceNormalization(
                axis=-1,
                center=False,
                scale=False,
                )(y)

            return add([skip, y])

        img = Input(shape=self.img_shape,)
        y = img
        y = conv7s1(y, self.ge_filters, False)
        y = downsample(y, self.ge_filters * 2)
        y = downsample(y, self.ge_filters * 4)
        for _ in range(self.residual):
            y = residual(y, self.ge_filters * 4)
        y = upsample(y, self.ge_filters * 2)
        y = upsample(y, self.ge_filters)
        y = conv7s1(y, 3, True)

        return Model(img, y)

    def build_discriminator(self):
        def conv4(layer, filters, strides=2, norm=True):
            y = Conv2D(
                filters=filters,
                kernel_size=4,
                strides=strides,
                padding='same',
                kernel_initializer=self.weight_init,
                )(layer)

            if norm:
                y = InstanceNormalization(
                    axis=-1,
                    center=False,
                    scale=False,
                    )(y)
            y = self.activation('leakyrelu', alpha=0.2)(y)

            return y

        img = Input(shape=self.img_shape)
        y = conv4(img, self.di_filters, norm=False)
        y = conv4(y, self.di_filters * 2)
        y = conv4(y, self.di_filters * 4)
        y = conv4(y, self.di_filters * 8, strides=1)

        y = Conv2D(
            filters=1,
            kernel_size=4,
            strides=1,
            padding='same',
            kernel_initializer=self.weight_init,
            )(y)

        return Model(img, y)

    def compile_model(self):
        self.di_A = self.build_discriminator()
        self.di_B = self.build_discriminator()
        self.di_A.compile(
            loss='mse',
            optimizer=self.optimizer(self.lr),
            metrics=['accuracy'],
            )
        self.di_B.compile(
            loss='mse',
            optimizer=self.optimizer(self.lr),
            metrics=['accuracy'],
            )

        self.ge_AB = self.build_generator()
        self.ge_BA = self.build_generator()

        self.di_A.trainable = False
        self.di_B.trainable = False

        imgs_A = Input(shape=self.img_shape)
        imgs_B = Input(shape=self.img_shape)
        fake_A = self.ge_BA(imgs_B)
        fake_B = self.ge_AB(imgs_A)

        reconst_A = self.ge_BA(fake_B)
        reconst_B = self.ge_AB(fake_A)

        imgs_A_id = self.ge_BA(imgs_A)
        imgs_B_id = self.ge_AB(imgs_B)

        real_A = self.di_A(fake_A)
        real_B = self.di_B(fake_B)

        self.combined = Model(
            inputs=[imgs_A, imgs_B],
            outputs=[real_A, real_B,
                    reconst_A, reconst_B,
                    imgs_A_id, imgs_B_id],
            )
        self.combined.compile(
            loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
            loss_weights=[
                self.lambda_valid,
                self.lambda_valid,
                self.lambda_reconst,
                self.lambda_reconst,
                self.lambda_id,
                self.lambda_id,
                ],
            optimizer=self.optimizer(self.lr),
            )

        self.di_A.trainable = True
        self.di_B.trainable = True

    def fit_generator(self, imgs_A, imgs_B, real):
        return self.combined.train_on_batch(
            [imgs_A, imgs_B],
            [real, real,
             imgs_A, imgs_B,
             imgs_A, imgs_B],
            )

    def fit_discriminator(self, imgs_A, imgs_B, real, fake):
        fake_A = self.ge_BA.predict(imgs_B)
        fake_B = self.ge_AB.predict(imgs_A)

        self.buf_A.append(fake_A)
        self.buf_B.append(fake_B)

        fake_A_rnd = random.sample(self.buf_A, min(len(self.buf_A), len(imgs_A)))
        fake_B_rnd = random.sample(self.buf_B, min(len(self.buf_B), len(imgs_B)))

        di_A_lss_real = self.di_A.train_on_batch(imgs_A, real)
        di_A_lss_fake = self.di_A.train_on_batch(fake_A_rnd, fake)

        di_B_lss_real = self.di_B.train_on_batch(imgs_B, real)
        di_B_lss_fake = self.di_B.train_on_batch(fake_B_rnd, fake)

        return di_A_lss_real, di_A_lss_fake, di_B_lss_real, di_B_lss_fake

    def fit(self, batch_size, max_epochs, show_every_n, folder, test_A, test_B):
        real = np.ones((batch_size, ) + self.disc_patch)
        fake = np.zeros((batch_size, ) + self.disc_patch)

        for epoch in range(self.epochs, max_epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.loader.load_batch()):
                di_A_lss_real, di_A_lss_fake, di_B_lss_real, di_B_lss_fake = \
                    self.fit_discriminator(imgs_A, imgs_B, real, fake)
                ge = self.fit_generator(imgs_A, imgs_B, real)

                print('Epoch %d / %d /// Batch %d / %d' %
                      (epoch, max_epochs, batch_i, self.loader.n_batches)
                      )
                print('D-A ||(ACC) Real %.4f | Fake %.4f (LOSS) | Real %.4f | Fake %.4f | Total %.4f' % 
                      (di_A_lss_real[1],
                       di_A_lss_fake[1],
                       di_A_lss_real[0],
                       di_A_lss_fake[0],
                       0.5 * (di_A_lss_real[0]+ di_A_lss_fake[0])
                       )
                      )
                print('D-B ||(ACC) Real %.4f | Fake %.4f (LOSS) | Real %.4f | Fake %.4f | Total %.4f' % 
                      (di_B_lss_real[1],
                       di_B_lss_fake[1],
                       di_B_lss_real[0],
                       di_B_lss_fake[0],
                       0.5 * (di_B_lss_real[0]+ di_B_lss_fake[0])
                       )
                      )
                print('G-A ||(LOSS) %.4f | Adv %.4f | Rec %.4f | ID %.4f' %
                      (ge[0],
                       np.sum(ge[1:3]),
                       np.sum(ge[3:5]),
                       np.sum(ge[5:7]),
                      )
                     )
                self.di_lss.append([di_A_lss_real, di_A_lss_fake, di_B_lss_real, di_B_lss_fake])
                self.di_lss.append(ge)
            
            if epoch % show_every_n == 0 or epoch + 1 == max_epochs:
                self.show_img(batch_i, folder, test_A, test_B)
                self.save_weights(file_name='weights_{}.h5'.format(epoch))

    def show_img(self, batch_i, folder, test_A, test_B):
        r, c = 2, 4
        for p in range(2):

            if p == 1:
                imgs_A = self.loader.load_data(domain="A", batch_size=1, is_testing=True)
                imgs_B = self.loader.load_data(domain="B", batch_size=1, is_testing=True)
            else:
                imgs_A = self.loader.load_img('data/%s/testA/%s' % (self.loader.dataset_name, test_A))
                imgs_B = self.loader.load_img('data/%s/testB/%s' % (self.loader.dataset_name, test_B))

            fake_B = self.g_AB.predict(imgs_A)
            fake_A = self.g_BA.predict(imgs_B)

            reconstr_A = self.g_BA.predict(fake_B)
            reconstr_B = self.g_AB.predict(fake_A)

            id_A = self.g_BA.predict(imgs_A)
            id_B = self.g_AB.predict(imgs_B)

            gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, id_A, imgs_B, fake_A, reconstr_B, id_B])

            gen_imgs = 0.5 * gen_imgs + 0.5
            gen_imgs = np.clip(gen_imgs, 0, 1)

            titles = ['Original', 'Translated', 'Reconstructed', 'ID']
            fig, axs = plt.subplots(r, c, figsize=(25,12.5))
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[cnt])
                    axs[i, j].set_title(titles[j])
                    axs[i,j].axis('off')
                    cnt += 1
            fig.savefig(os.path.join(folder ,"images/%d_%d_%d.png" % (p, self.epoch, batch_i)))
            plt.close()

    def plot_loss(self):
        fig = plt.figure(figsize=(200, 50))
        ax1 = fig.add_subplot(111)
        ax1.plot(len(self.di_lss), self.di_real_lss)
        ax2 = fig.add_subplot(121)
        ax2.plot(len(self.di_fake_lss), self.di_fake_lss)
        ax3 = fig.add_subplot(131)
        ax3.plot(len(self.di_lss), self.di_lss)
        ax4 = fig.add_subplot(141)
        ax4.plot(len(self.ge_lss), self.ge_lss)

        plt.show()
        plt.cla()
        plt.clf()
