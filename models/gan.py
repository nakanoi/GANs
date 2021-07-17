import matplotlib.pyplot as plt

from tensorflow.keras.layers import (Input, Reshape, Dense, Flatten,
                                     BatchNormalization, Dropout,
                                     )
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model

from models.basemodel import BaseModel, DataLoader, np, os


class GAN(BaseModel):
    def __init__(self,
                 input_dim = (28, 28, 1),
                 di_neurons = (512, 256, 1),
                 di_batch_norm = 0.8,
                 di_dropout = None,
                 di_active = 'leakyrelu',
                 di_alpha = 0.2,
                 di_lr = 0.0002,
                 ge_neurons = (256, 512, 1024),
                 ge_batch_norm = 0.8,
                 ge_dropout = None,
                 ge_active = 'leakyrelu',
                 ge_alpha = 0.2,
                 ge_lr = 0.0002,
                 optimizer = 'adam',
                 beta1 = 0.5,
                 z_dim = 100,
                 weight_init = (0, 0.02),
                 start_epoch = 0,
                 k = 1,
                 loader = mnist,
                 data_name = '',
                 ):
        super().__init__()

        self.input_dim = input_dim

        self.di_neurons = di_neurons
        self.di_batch_norm = di_batch_norm
        self.di_dropout = di_dropout
        self.di_active = di_active
        self.di_alpha = di_alpha
        self.di_lr = di_lr

        self.ge_neurons = ge_neurons
        self.ge_batch_norm = ge_batch_norm
        self.ge_dropout = ge_dropout
        self.ge_active = ge_active
        self.ge_alpha = ge_alpha
        self.ge_lr = ge_lr

        self.opt = optimizer
        self.beta1 = beta1
        self.z_dim = z_dim
        self.k = k
        if weight_init is None:
            weight_init = (0, 1)
        self.weight_init = RandomNormal(mean=weight_init[0], stddev=weight_init[1])

        self.epochs = start_epoch
        self.k = k
        self.data_name = data_name

        if loader is None:
            self.loader = DataLoader(self.data_name, tuple(input_dim[:2]))
        else:
            self.loader = loader

        self.di_len = len(di_neurons)
        self.ge_len = len(ge_neurons)
        self.di_real_lss = []
        self.di_fake_lss = []
        self.di_lss = []
        self.ge_lss = []

        self.build_generator()
        self.build_discriminator()
        self.build_adversal()

    def build_generator(self):
        ge_input = Input(shape=(self.z_dim, ), name='G_input')
        x = ge_input

        for i in range(self.ge_len):
            x = Dense(units=self.ge_neurons[i],
                      kernel_initializer=self.weight_init)(x)
            x = BatchNormalization(momentum=self.ge_batch_norm)(x)
            x = self.activation(self.ge_active, alpha=self.ge_alpha)(x)

            if self.ge_dropout is not None:
                x = Dropout(self.ge_dropout)(x)

        x = Dense(np.prod(self.input_dim))(x)
        x = self.activation('tanh')(x)
        x = Reshape(self.input_dim)(x)

        ge_output = x
        self.generator = Model(ge_input, ge_output)

        setattr(self.generator, 'model_name', 'generator')
        self.models.setdefault('Generator', self.generator)

    def build_discriminator(self):
        di_input = Input(shape=self.input_dim, name='D_input')
        x = di_input
        x = Flatten()(x)

        for i in range(self.di_len):
            x = Dense(units=self.di_neurons[i],
                      kernel_initializer=self.weight_init)(x)
            x = BatchNormalization(momentum=self.di_batch_norm)(x)

            if i == self.di_len - 1:
                x = self.activation('sigmoid')(x)
            else:
                x = self.activation(self.di_active, alpha=self.di_alpha)(x)
                if self.di_dropout is not None:
                    x = Dropout(rate=self.di_dropout)(x)

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
        self.combined.compile(loss='binary_crossentropy',
                           optimizer=self.optimizer(self.ge_lr),
                           metrics=['accuracy'],
                           )

        setattr(self.combined, 'model_name', 'combined')
        self.models.setdefault('Full Model', self.combined)
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

    def fit(self, batch_size, max_epochs, show_every_n):
        (x_train, _), (_, _) = self.loader.load_data()

        for epoch in range(max_epochs):
            for j in range(self.k):
                di_hist_real, di_hist_fake = self.fit_discriminator(x_train,
                                                                    batch_size,
                                                                    )
                self.di_real_lss.append(di_hist_real.history['loss'][0])
                self.di_fake_lss.append(di_hist_fake.history['loss'][0])
                self.di_lss.append(0.5 * (di_hist_real.history['loss'][0] + 
                                      di_hist_fake.history['loss'][0]))

            ge_hist = self.fit_generator(batch_size)
            self.ge_lss.append(ge_hist.history['loss'][0])

            if epoch % show_every_n == 0 or epoch == max_epochs - 1:
                print('|D|Total %.4f, Real %.4f, Fake %.4f\n|G|%.4f' %
                      (self.di_lss[-1],
                       self.di_real_lss[-1],
                       self.di_real_lss[-1],
                       self.ge_lss[-1])
                      )
                self.show_img(3, epoch, './imgs', 'sample{}.png'.format(epoch))
                self.save_weights(file_name='weights_{}.h5'.format(epoch))
        self.plot_loss()

    def show_img(self, img_num, epoch, folder, file_name='image.ong'):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, file_name)

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

        fig.savefig(path)
        plt.show()
        plt.cla()
        plt.clf()

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

