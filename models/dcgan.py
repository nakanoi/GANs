import matplotlib.pyplot as plt

from tensorflow.keras.layers import (Input, Reshape, Dense, Conv2D,
                                     BatchNormalization, UpSampling2D,
                                     Dropout, Flatten, Conv2DTranspose,
                                     )
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model

from models.basemodel import BaseModel, DataLoader, np, os, datetime


class DCGAN(BaseModel):
    def __init__(self,
                 input_dim = (28, 28, 1),
                 di_conv_filters = [64, 128, 512, 1024],
                 di_conv_kernels = [5, 5, 5, 5],
                 di_conv_strides = [2, 2, 2, 1],
                 di_batch_norm = None,
                 di_dropout = 0.2,
                 di_active = 'leakyrelu',
                 alpha = 0.2,
                 di_lr = 0.0008,
                 ge_initial_size = (7, 7, 64),
                 ge_upsample = [2, 2, 1, 1],
                 ge_conv_filters = [512, 256, 128, 1],
                 ge_conv_kernels = [5, 5, 5, 5],
                 ge_conv_strides = [1, 1, 1, 1],
                 ge_batch_norm = None,
                 ge_active = 'relu',
                 ge_lr = 0.0004,
                 optimizer = 'adam',
                 beta1 = 0.5,
                 z_dim = 100,
                 weight_init = [0, 0.02],
                 start_epoch = 0,
                 k = 1,
                 ID = 0,
                 loader = None,
                 data_name = '',
                 color = 'gray',
                 ):
        self.input_dim = input_dim

        self.di_conv_filters = di_conv_filters
        self.di_conv_kernels = di_conv_kernels
        self.di_conv_strides = di_conv_strides
        self.di_batch_norm = di_batch_norm
        self.di_dropout = di_dropout
        self.di_active = di_active
        self.alpha = alpha
        self.di_lr = di_lr

        self.ge_initial_size = ge_initial_size
        self.ge_upsample = ge_upsample
        self.ge_conv_filters = ge_conv_filters
        self.ge_conv_kernels = ge_conv_kernels
        self.ge_conv_strides = ge_conv_strides
        self.ge_batch_norm = ge_batch_norm
        self.ge_active = ge_active
        self.ge_lr = ge_lr

        self.opt = optimizer
        self.beta1 = beta1
        self.z_dim = z_dim
        if weight_init is None:
            weight_init = [0, 1]
        self.weight_init = RandomNormal(*weight_init)
        self.color = color

        self.epochs = start_epoch
        self.k = k
        self.data_name = data_name

        if loader is None:
            self.loader = DataLoader(self.data_name,
                                     ID,
                                     shape=tuple(input_dim[:2]),
                                     color=color,
                                     section='DCGAN')
        else:
            self.loader = loader

        self.di_len = len(di_conv_filters)
        self.ge_len = len(ge_conv_filters)

        super().__init__()
        self.build_generator()
        self.build_discriminator()
        self.build_adversal()

    def build_generator(self):
        ge_input = Input(shape=(self.z_dim, ), name='G_input')
        x = ge_input

        x = Dense(np.prod(self.ge_initial_size),
                  kernel_initializer=self.weight_init)(x)

        if self.ge_batch_norm is not None:
            x = BatchNormalization(momentum=self.ge_batch_norm)(x)

        x = self.activation(self.ge_active, self.alpha)(x)
        x = Reshape(self.ge_initial_size)(x)

        for i in range(self.ge_len):
            if self.ge_upsample[i] == 2:
                x = UpSampling2D()(x)
                x = Conv2D(filters=self.ge_conv_filters[i],
                           kernel_size=self.ge_conv_kernels[i],
                           strides=self.ge_conv_strides[i],
                           padding='same',
                           kernel_initializer=self.weight_init,
                           name='G_conv_' + str(i),
                           )(x)
            else:
                x = Conv2DTranspose(filters=self.ge_conv_filters[i],
                                    kernel_size=self.ge_conv_kernels[i],
                                    strides=self.ge_conv_strides[i],
                                    padding='same',
                                    kernel_initializer=self.weight_init,
                                    name='G_conv_' + str(i),
                                    )(x)

            if i < self.ge_len - 1:
                if self.ge_batch_norm is not None:
                    x = BatchNormalization(momentum=self.ge_batch_norm)(x)
                x = self.activation(self.ge_active, self.alpha)(x)
            else:
                x = self.activation('tanh')(x)

        ge_output = x
        self.generator = Model(ge_input, ge_output)

        setattr(self.generator, 'model_name', 'generator')
        self.models.setdefault('Generator', self.generator)

    def build_discriminator(self):
        di_input = Input(shape=self.input_dim, name='D_input')
        x = di_input

        for i in range(self.di_len):
            x = Conv2D(
                filters=self.di_conv_filters[i],
                kernel_size=self.di_conv_kernels[i],
                strides=self.di_conv_strides[i],
                padding='same',
                kernel_initializer=self.weight_init,
                name='D_conv_' + str(i),
                )(x)

            if i > 0 and self.di_batch_norm is not None:
                x = BatchNormalization(momentum=self.di_batch_norm)(x)

            x = self.activation(self.di_active, self.alpha)(x)

            if self.di_dropout is not None:
                x = Dropout(rate=self.di_dropout)(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid',
                  kernel_initializer=self.weight_init)(x)
        di_output = x
        self.discriminator = Model(di_input, di_output)

        setattr(self.discriminator, 'model_name', 'generator')
        self.models.setdefault('Discriminator', self.discriminator)

    def build_adversal(self):
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer(self.di_lr),
            metrics=['accuracy'],
            )
        self.trainable(self.discriminator, False)

        model_input = Input(shape=(self.z_dim, ), name='model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.combined = Model(model_input, model_output)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer(self.ge_lr),
            metrics=['accuracy'],
            experimental_run_tf_function=False,
            )

        setattr(self.combined, 'model_name', 'combined')
        self.models.setdefault('Whole_Model', self.combined)
        self.trainable(self.discriminator, True)

    def train_generator(self, batch_size):
        real = np.ones((batch_size, 1))
        z = np.random.normal(0, 1, (batch_size, self.z_dim))

        return self.combined.train_on_batch(z, real)

    def train_discriminator(self, x_train, batch_size):
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        idcs = np.random.randint(0, x_train.shape[0], batch_size)
        real_imgs = x_train[idcs]

        z = np.random.normal(0, 1, (batch_size, self.z_dim))
        fake_imgs = self.generator.predict(z)

        di_real = self.discriminator.train_on_batch(real_imgs, real)
        di_fake = self.discriminator.train_on_batch(fake_imgs, fake)
        di_lss = (di_real[0] + di_fake[0]) / 2
        di_acc = (di_real[1] + di_fake[1]) / 2

        return di_real, di_fake, di_lss, di_acc

    def train(self,
              batch_size,
              max_epochs,
              show_every_n,
              train_data=None,
              train_batch=False):
        s = datetime.now()
        if train_batch:
            for epoch in range(max_epochs):
                for batch_i, (x_train, y_train) in enumerate(self.loader.load_batch()): 
                    for j in range(self.k):
                        di = self.train_discriminator(x_train, batch_size)
                    ge = self.train_generator(batch_size)

                    self.di_real_lss.append(di[0][0])
                    self.di_fake_lss.append(di[1][0])
                    self.di_lss.append(di[2])
                    self.di_acc.append(di[3] * 100)
                    self.ge_lss.append(ge[0])

                if epoch % show_every_n == 0:
                    elapsed_time = datetime.now() - s
                    print ('[Epochs %d/%d] [Batch %d/%d] [D Acc %.2f Total %.4f Real %.4f Fake %.4f] [G %.4f] Time %s' %
                           (epoch, max_epochs,
                            batch_i, self.loader.n_batches,
                            di[3] * 100, di[2], di[0][0], di[1][0], ge[0],
                            elapsed_time))

                    self.show_img(self.generator, self.z_dim, 'sample_{}.png'.format(epoch), color='gray')
                    self.save_weights(file_name='weights_{}.h5'.format(epoch))
                    self.save_models(epoch=epoch)

            self.show_img(self.generator, self.z_dim, 'sample_last.png', color='gray')
            self.save_models()
            self.plot_loss()

        else:
            if len(train_data) == 1:
                x_train = train_data[0]
            else:
                x_train, t_train = train_data

            for epoch in range(max_epochs):
                for j in range(self.k):
                    di = self.train_discriminator(x_train, batch_size)
                ge = self.train_generator(batch_size)
    
                self.di_real_lss.append(di[0][0])
                self.di_fake_lss.append(di[1][0])
                self.di_lss.append(di[2])
                self.di_acc.append(di[3] * 100)
                self.ge_lss.append(ge[0])

                if epoch % show_every_n == 0:
                    elapsed_time = datetime.now() - s
                    print ('[Epochs %d/%d] [D Acc %.2f Total %.4f Real %.4f Fake %.4f] [G %.4f] Time %s' %
                           (epoch, max_epochs,
                            di[3] * 100, di[2], di[0][0], di[1][0], ge[0],
                            elapsed_time))

                    self.show_img(3, epoch, file_name='sample_{}.png'.format(epoch))
                    self.save_weights(file_name='weights_{}.h5'.format(epoch))
                    self.save_models(epoch=epoch)

            self.show_img(self.generator, self.z_dim, 'sample_last.png', color='gray')
            self.save_models()
            self.plot_loss()
