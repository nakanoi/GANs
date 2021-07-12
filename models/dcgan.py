from tensorflow.keras.layers import (Input, Reshape, Dense, Conv2D,
                                     BatchNormalization, UpSampling2D,
                                     Dropout, Flatten, Conv2DTranspose,
                                     )
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model

from GANGANGAN.models.basemodel import BaseModel, plt, np, os


class DCGAN(BaseModel):
    def __init__(self,
                 input_dim,
                 di_conv_filters=[64, 128, 512, 1024],
                 di_conv_kernels=[5, 5, 5, 5],
                 di_conv_strides=[2, 2, 2, 1],
                 di_batch_norm=None,
                 di_dropout=0.2,
                 di_active='leakrelu',
                 alpha=0.2,
                 di_lr=0.0008,
                 ge_initial_size=(4, 4, 1024),
                 ge_upsample=[2, 2, 1, 1],
                 ge_conv_filters=[512, 256, 128, 3],
                 ge_conv_kernels=[5, 5, 5, 5],
                 ge_conv_strides=[1, 1, 1, 1],
                 ge_batch_norm=None,
                 ge_active='relu',
                 ge_lr=0.0004,
                 optimizer='adam',
                 beta1=0.5,
                 z_dim=100,
                 weight_init=(0, 0.02),
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
        self.weight_init = RandomNormal(mean=weight_init[0], stddev=weight_init[1])

        self.di_len = len(di_conv_filters)
        self.ge_len = len(ge_conv_filters)
        self.di_real_lss = []
        self.di_fake_lss = []
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
        x = BatchNormalization(momentum=self.ge_batch_norm)(x)
        x = self.activation(self.ge_active)(x)
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
                x = BatchNormalization(momentum=self.ge_batch_norm)(x)
                x = self.activation(self.ge_active)(x)
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
            x = Conv2D(filters=self.di_conv_filters[i],
                       kernel_size=self.di_conv_kernels[i],
                       strides=self.di_conv_strides[i],
                       padding='same',
                       kernel_initializer=self.weight_init,
                       name='D_conv_' + str(i),
                       )(x)
            x = BatchNormalization(momentum=self.di_batch_norm)(x)
            x = self.activation(self.di_active, self.alpha)(x)
            x = Dropout(rate=self.di_dropout)(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid',
                  kernel_initializer=self.weight_init)(x)
        di_output = x
        self.discriminator = Model(di_input, di_output)

        setattr(self.discriminator, 'model_name', 'generator')
        self.models.setdefault('Discriminator', self.discriminator)

    def build_adversal(self):
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.optimizer(self.di_lr),
                                   metrics=['accuracy'],
                                   )
        self.trainable(self.discriminator, False)

        model_input = Input(shape=(self.z_dim, ), name='model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.combined = Model(model_input, model_output)

        setattr(self.combined, 'model_name', 'combined')
        self.models.setdefault('Full Model', self.combined)

        self.combined.compile(loss='binary_crossentropy',
                           optimizer=self.optimizer(self.ge_lr),
                           metrics=['accuracy'],
                           experimental_run_tf_function=False,
                           )

        self.trainable(self.discriminator, True)

    def fit_generator(self, batch_size):
        real = np.ones((batch_size, 1))
        z = np.random.normal(0, 1, (batch_size, self.z_dim))
        ge_hist = self.combined.fit(z,
                                   real,
                                   batch_size=batch_size,
                                   epochs=1,
                                   shuffle=True,
                                   )
        return ge_hist

    def fit_discriminator(self, x_train, batch_size):
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        z = np.random.normal(0, 1, (batch_size, self.z_dim))
        idcs = np.random.randint(0, len(x_train), batch_size)
        real_imgs = x_train[idcs]
        fake_imgs = self.generator.predict(z)

        di_hist_real = self.discriminator.fit(real_imgs,
                                              real,
                                              batch_size=batch_size,
                                              epochs=1,
                                              shuffle=True,
                                              )
        di_hist_fake = self.discriminator.fit(fake_imgs,
                                              fake,
                                              batch_size=batch_size,
                                              epochs=1,
                                              shuffle=True,
                                              )
        return di_hist_real, di_hist_fake

    def fit(self, x_train, batch_size, max_epochs):
        for epoch in range(max_epochs):
            di_hist_real, di_hist_fake = self.fit_discriminator(x_train,
                                                                batch_size,
                                                                )
            ge_hist = self.fit_generator(batch_size)
            self.di_real_lss.append(di_hist_real.history['loss'])
            self.di_fake_lss.append(di_hist_fake.history['loss'])
            self.ge_lss.append(ge_hist.history['loss'])
            self.show_img(3, epoch, '.', 'sample{}.png'.format(epoch))

    def show_img(self, img_num, epoch, folder, file_name='image.ong'):
        fig, axes = plt.subplots(1,
                                 img_num,
                                 figsize=(self.input_dim[0] * img_num,
                                          self.input_dim[1] + 4,
                                          )
                                 )
        plt.rcParams["font.size"] = 60
        plt.tight_layout()
        z = np.random.normal(0, 1, (img_num, self.z_dim))
        fake_imgs = self.generator.predict(z)

        for i in range(img_num):
            axes[i].imshow(fake_imgs[i].reshape(self.input_dim), cmap='gray')
            axes[i].set_title('Epoch : {}'.format(epoch))
            axes[i].axis('off')

        fig.savefig(os.path.join(folder, file_name))
        plt.show()
        plt.cla()
        plt.clf()

    def plot_loss(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(len(self.di_real_lss), self.di_real_lss)
        ax2 = fig.add_subplot(121)
        ax2.plot(len(self.di_fake_lss), self.di_fake_lss)
        ax3 = fig.add_subplot(131)
        ax3.plot(len(self.ge_lss), self.ge_lss)
        plt.show()
        plt.cla()
        plt.clf()

