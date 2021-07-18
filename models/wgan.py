from tensorflow.keras.layers import (Input, Reshape, Dense, Conv2D,
                                     BatchNormalization, UpSampling2D,
                                     Dropout, Flatten, Conv2DTranspose,
                                     )
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from models.basemodel import BaseModel, DataLoader, np


class WGAN(BaseModel):
    def __init__(self,
                 input_dim = (28, 28, 1),
                 di_conv_filters = [64, 128, 512, 1024],
                 di_conv_kernels = [5, 5, 5, 5],
                 di_conv_strides = [2, 2, 2, 1],
                 di_batch_norm = None,
                 di_dropout = None,
                 di_active = 'leakyrelu',
                 di_lr = 0.00005,
                 ge_initial_size = (7, 7, 64),
                 ge_upsample = [2, 2, 1, 1],
                 ge_conv_filters = [512, 256, 128, 1],
                 ge_conv_kernels = [5, 5, 5, 5],
                 ge_conv_strides = [1, 1, 1, 1],
                 ge_batch_norm = 0.8,
                 ge_dropout = None,
                 ge_active = 'leakyrelu',
                 ge_lr = 0.00005,
                 optimizer = 'rmsprop',
                 beta1 = 0.5,
                 z_dim = 100,
                 weight_init = (0, 0.02),
                 start_epoch = 0,
                 k = 5,
                 loader = None,
                 data_name = 'mnist',
                 ID = 0,
                 color = 'gray',
                 ):
        self.input_dim = input_dim
        self.di_conv_filters = di_conv_filters
        self.di_conv_kernels = di_conv_kernels
        self.di_conv_strides = di_conv_strides
        self.di_batch_norm = di_batch_norm
        self.di_dropout = di_dropout
        self.di_active = di_active
        self.di_lr = di_lr

        self.ge_initial_size = ge_initial_size
        self.ge_upsample = ge_upsample
        self.ge_conv_filters = ge_conv_filters
        self.ge_conv_kernels = ge_conv_kernels
        self.ge_conv_strides = ge_conv_strides
        self.ge_batch_norm = ge_batch_norm
        self.ge_dropout = ge_dropout
        self.ge_active = ge_active
        self.ge_lr = ge_lr

        self.opt = optimizer
        self.beta1 = beta1
        self.z_dim = z_dim

        if weight_init is None:
            weight_init = (0, 1)
        self.weight_init = RandomNormal(mean=weight_init[0],
                                        stddev=weight_init[1])

        self.epochs = start_epoch
        self.k = k
        self.data_name = data_name

        if loader is None:
            self.loader = DataLoader(data_name, ID, shape=input_dim, color=color, section='WGAN')
        self.color = color

        self.di_len = len(di_conv_filters)
        self.ge_len = len(ge_conv_filters)
        self.di_real_lss = []
        self.di_fake_lss = []
        self.di_lss = []
        self.ge_lss = []

        super().__init__()
        self.build_gengerator()
        self.build_discriminator()
        self.build_adversarial()

    def wassarstein(self, t, y):
        return -K.mean(t * y)

    def build_gengerator(self):
        ge_input = Input(shape=(self.z_dim, ), name='Generator_input')
        x = ge_input

        x = Dense(np.prod(self.ge_initial_size),
                  kernel_initializer=self.weight_init
                  )(x)
        if self.ge_batch_norm is not None:
            x = BatchNormalization(momentum=self.ge_batch_norm)(x)

        x = self.activation(self.ge_active)(x)
        x = Reshape(target_shape=self.ge_initial_size)(x)

        if self.ge_dropout is not None:
            x = Dropout(self.ge_dropout)(x)

        for i in range(self.ge_len):
            if self.ge_upsample[i] == 2:
                x = UpSampling2D()(x)
                x = Conv2D(
                    filters=self.ge_conv_filters[i],
                    kernel_size=self.ge_conv_kernels[i],
                    strides=self.ge_conv_strides[i],
                    padding='same',
                    name='ge_conv_{}'.format(i),
                    kernel_initializer=self.weight_init
                    )(x)
            else:
                x = Conv2DTranspose(
                    filters=self.ge_conv_filters[i],
                    kernel_size=self.ge_conv_kernels[i],
                    strides=self.ge_conv_strides[i],
                    padding='same',
                    kernel_initializer=self.weight_init,
                    name='ge_conv2Trans_{}'.format(i)
                    )(x)

            if i < self.ge_len - 1:
                if self.ge_batch_norm is not None:
                    x = BatchNormalization()(x)
                x = self.activation(self.ge_active)(x)
            else:
                x = self.activation('tanh')(x)

        self.generator = Model(ge_input, x)

    def build_discriminator(self):
        di_input = Input(shape=self.input_dim)
        x = di_input

        for i in range(self.di_len):
            x = Conv2D(
                filters=self.di_conv_filters[i],
                kernel_size=self.di_conv_kernels[i],
                strides=self.di_conv_strides[i],
                padding='same',
                kernel_initializer=self.weight_init,
                name='di_conv_{}'.format(i)
                )(x)
            if self.di_batch_norm is not None:
                print(self.di_batch_norm)
                x = BatchNormalization(momentum=self.di_batch_norm)(x)

            x = self.activation(self.di_active)(x)

            if self.di_dropout is not None:
                x = Dropout(self.di_dropout)(x)
        
        x = Flatten()(x)
        x = Dense(1, activation=None, kernel_initializer=self.weight_init)(x)

        self.discriminator = Model(di_input, x)


    def build_adversarial(self):
        self.discriminator.compile(
            optimizer=self.optimizer(self.di_lr),
            loss=self.wassarstein,
            metrics=['accuracy']
            )

        self.trainable(self.discriminator, False)

        combined_input = Input(shape=(self.z_dim, ))
        combined_output = self.discriminator(self.generator(combined_input))
        self.combined = Model(combined_input, combined_output)
        self.combined.compile(
            optimizer=self.optimizer(self.ge_lr),
            loss=self.wassarstein,
            metrics=['accuracy']
            )

        self.trainable(self.discriminator, True)

        self.models.setdefault('Discriminator', self.discriminator)
        self.models.setdefault('Generator', self.generator)
        self.models.setdefault('Combined', self.combined)

    def train_generator(self, batch_size):
        real = np.ones((batch_size, 1))
        z = np.random.normal(0, 1, (batch_size, self.z_dim))
        ge_lss = self.combined.train_on_batch(z, real)

        return np.array(ge_lss)

    def train_discriminator(self, x_train, batch_size, clip):
        real = np.ones((batch_size, 1))
        fake = -np.ones((batch_size, 1))

        idcs = np.random.randint(0, x_train.shape[0], batch_size)
        real_imgs = x_train[idcs]
        di_real = self.discriminator.train_on_batch(real_imgs, real)

        z = np.random.normal(0, 1, (batch_size, self.z_dim))
        fake_imgs = self.generator.predict(z)
        di_fake = self.discriminator.train_on_batch(fake_imgs, fake)

        di_real, di_fake = np.array(di_real), np.array(di_fake)

        di_tot = (di_real + di_fake) / 2

        for l in self.discriminator.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -clip, clip) for w in weights]
            l.set_weights(weights)

        return di_real, di_fake, di_tot

    def train(self, batch_size, max_epochs, show_every_n, clip=0.01):
        x_train, _ = self.loader.load_np_data(self.data_name)

        for epoch in range(self.epochs, max_epochs):
            for k in range(self.k):
                di_lss = self.train_discriminator(x_train, batch_size, clip)
            ge_lss = self.train_generator(batch_size)

            di_real, di_fake, di_lss = di_lss

            print('%d/%d [D :Loss = %.4f, Real = %.4f, Fake = %.4f, Acc = %.2f] [G :Loss = %.4f]' %
                  (epoch, max_epochs,
                   di_lss[0], di_real[0], di_fake[0], di_lss[1],
                   ge_lss[0])
                  )

            self.di_real_lss.append(di_real[0])
            self.di_fake_lss.append(di_fake[0])
            self.di_lss.append(di_lss[0])
            self.ge_lss.append(ge_lss[0])

            if epoch % show_every_n:
                self.show_img(self.generator, epoch, self.z_dim,
                              self.loader.folder, color=self.color)

        self.save_models(self.loader.folder)
        self.show_img()

