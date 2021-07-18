import matplotlib.pyplot as plt

from tensorflow.keras.layers import (Input, Reshape, Dense, Conv2D,
                                     BatchNormalization, UpSampling2D,
                                     Dropout, Flatten, Conv2DTranspose,
                                     )
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model

from models.basemodel import BaseModel, DataLoader, np, os


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
                 weight_init = (0, 0.02),
                 start_epoch = 0,
                 k = 1,
                 loader = None,
                 data_name = '',
                 ID = 1,
                 color = 'gray'
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
            weight_init = (0, 1)
        self.weight_init = RandomNormal(mean=weight_init[0], stddev=weight_init[1])

        self.epochs = start_epoch
        self.k = k
        self.data_name = data_name

        if loader is None:
            self.loader = DataLoader(self.data_name, ID, tuple(input_dim[:2]), color)
        else:
            self.loader = loader

        self.di_len = len(di_conv_filters)
        self.ge_len = len(ge_conv_filters)
        self.di_real_lss = []
        self.di_fake_lss = []
        self.di_lss = []
        self.ge_lss = []

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

    def train(self, batch_size, max_epochs, show_every_n):
        x_train, _ = self.loader.load_np_data(self.data_name)

        for epoch in range(max_epochs):
            di = self.train_discriminator(x_train, batch_size)
            ge = self.train_generator(batch_size)
            print ("%d [D loss: (%.3f)(R %.3f, F %.3f)] [D acc: (%.3f)(%.3f, %.3f)]\n[G loss: %.3f] [G acc: %.3f]"
                   % (epoch, di[2], di[0][0], di[1][0],
                      di[3], di[0][1], di[1][1], ge[0], ge[1]))

            self.di_real_lss.append(di[0][0])
            self.di_fake_lss.append(di[1][0])
            self.di_lss.append(di[2])
            self.ge_lss.append(ge)

            if epoch % show_every_n == 0 or epoch == max_epochs - 1:
                self.show_img(3, epoch, file_name='sample_{}.png'.format(epoch))
        self.save_models(self.loader.folder)

    def show_img(self, img_num, epoch, file_name='sample.png', show=False):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c, figsize=(15,15))
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(np.squeeze(gen_imgs[cnt, :, :, :]), cmap =self.color)
                axs[i,j].axis('off')
                cnt += 1
        if show:
            plt.show()
        fig.savefig(os.path.join(self.loader.folder, 'images', file_name))
        plt.close()

    def plot_loss(self):
        fig = plt.figure(figsize=(200, 50))
        ax1 = fig.add_subplot(111)
        ax1.plot(len(self.di_real_lss), self.di_real_lss)
        ax2 = fig.add_subplot(121)
        ax2.plot(len(self.di_fake_lss), self.di_fake_lss)
        ax3 = fig.add_subplot(131)
        ax3.plot(len(self.di_lss), self.di_lss)
        ax4 = fig.add_subplot(141)
        ax4.plot(len(self.ge_lss), self.ge_lss)
        plt.show()
        plt.cla()
        plt.clf()

