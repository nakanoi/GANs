from tensorflow.keras.layers import (Input, Reshape, Dense, Conv2D, Layer,
                                     BatchNormalization, UpSampling2D,
                                     Dropout, Flatten, Conv2DTranspose,
                                     )
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from models.basemodel import BaseModel, DataLoader, np, datetime, plt
from functools import partial


class RandomWeightAverage(Layer):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def get_config(self):
        config = {'batch_size': self.batch_size}
        base_config = super().get_config()
 
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return alpha * inputs[0] + (1 - alpha) * inputs[1]


class WGANGP(BaseModel):
    def __init__(self,
                 input_dim = (28, 28, 1),
                 di_conv_filters = [64, 128, 512, 1024],
                 di_conv_kernels = [5, 5, 5, 5],
                 di_conv_strides = [2, 2, 2, 1],
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
                 grad_weights = 10,
                 start_epoch = 0,
                 k = 5,
                 loader = None,
                 data_name = 'mnist',
                 ID = 0,
                 color = 'gray',
                 batch_size = 64
                 ):
        self.input_dim = input_dim
        self.di_conv_filters = di_conv_filters
        self.di_conv_kernels = di_conv_kernels
        self.di_conv_strides = di_conv_strides
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

        self.grad_weights = grad_weights
        self.epochs = start_epoch
        self.k = k
        self.data_name = data_name

        if loader is None:
            self.loader = DataLoader(data_name, ID, shape=input_dim, color=color, section='WGANGP')
        self.color = color
        self.batch_size = batch_size

        self.di_len = len(di_conv_filters)
        self.ge_len = len(ge_conv_filters)
        self.di_sample = []

        super().__init__()
        self.build_gengerator()
        self.build_discriminator()
        self.build_adversarial()

    def wassarstein(self, real, pred):
        return -K.mean(real * pred)

    def gradinet_penalty_loss(self, real, pred, samples):
        grads = K.gradients(pred, samples)[0]
        grad_l2_norm = K.sqrt(K.sum(K.square(grads), axis=np.arange(1, len(grads.shape))))
        grad_penalty = K.square(1 - grad_l2_norm)

        return K.mean(grad_penalty)

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
                    x = BatchNormalization(momentum=self.ge_batch_norm)(x)
                x = self.activation(self.ge_active)(x)
            else:
                x = self.activation('tanh')(x)

        self.generator = Model(ge_input, x)

    def build_discriminator(self):
        di_input = Input(shape=self.input_dim, name='Discriminator_input')
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

            x = self.activation(self.di_active)(x)

            if self.di_dropout is not None:
                x = Dropout(self.di_dropout)(x)
        
        x = Flatten()(x)
        x = Dense(1, activation=None, kernel_initializer=self.weight_init)(x)

        self.discriminator = Model(di_input, x)


    def build_adversarial(self):
        self.trainable(self.generator, False)

        real_imgs = Input(shape=self.input_dim, )
        z = Input(shape=(self.z_dim, ), )
        fake_imgs = self.generator(z)

        real = self.discriminator(real_imgs)
        fake = self.discriminator(fake_imgs)

        samples_imgs = RandomWeightAverage(self.batch_size)([real_imgs, fake_imgs])
        samples = self.discriminator(samples_imgs)

        gp_lss = partial(self.gradinet_penalty_loss,
                         samples=samples)
        gp_lss.__name__ = 'gradient_penalty'

        self.discriminator_model = Model(inputs=[real_imgs, z],
                                         outputs=[real, fake, samples])
        self.discriminator_model.compile(
            loss=[self.wassarstein, self.wassarstein, gp_lss],
            optimizer=self.optimizer(self.di_lr),
            loss_weights=[1, 1, self.grad_weights],
            experimental_run_tf_function=False
            )

        self.trainable(self.generator, True)
        self.trainable(self.discriminator, False)

        combined_input = Input(shape=(self.z_dim, ))
        combined_output = self.discriminator(self.generator(combined_input))
        self.combined = Model(combined_input, combined_output)
        self.combined.compile(
            optimizer=self.optimizer(self.ge_lr),
            loss=self.wassarstein,
            metrics=['accuracy'],
            experimental_run_tf_function=False
            )

        self.trainable(self.discriminator, True)

        self.models.setdefault('Discriminator', self.discriminator)
        self.models.setdefault('Generator', self.generator)
        self.models.setdefault('Whole_Discriminator', self.discriminator_model)
        self.models.setdefault('Combined', self.combined)

    def train_generator(self):
        real = np.ones((self.batch_size, 1))
        z = np.random.normal(0, 1, (self.batch_size, self.z_dim))

        return self.combined.train_on_batch(z, real)

    def train_discriminator(self, x_train, clip):
        real = np.ones((self.batch_size, 1))
        fake = -np.ones((self.batch_size, 1))
        zeros = np.zeros((self.batch_size, 1))

        idcs = np.random.randint(0, x_train.shape[0], self.batch_size)
        real_imgs = x_train[idcs]
        z = np.random.normal(0, 1, (self.batch_size, self.z_dim))

        di_tot, di_real, di_fake, di_mid = self.discriminator_model.train_on_batch([real_imgs, z], [real, fake, zeros])

        return di_tot, di_real, di_fake, di_mid

    def train(self,
              max_epochs,
              show_every_n,
              clip=0.01,
              train_data=None,
              train_batch=False):
        s = datetime.now()
        if train_batch:
            for epoch in range(self.epochs, max_epochs):
                for batch_i, (x_train, y_train) in enumerate(self.loader.load_batch()): 
                    for k in range(self.k):
                        di = self.train_discriminator(x_train, clip)
                    ge = self.train_generator()

                    self.di_real_lss.append(di[1])
                    self.di_fake_lss.append(di[2])
                    self.di_lss.append(di[0])
                    self.di_sample.append(di[3])
                    self.ge_lss.append(ge[0])

                if epoch % show_every_n == 0:
                    elapsed_time = datetime.now() - s
                    print ('[Epochs %d/%d] [Batch %d/%d] [D Total %.4f Real %.4f Fake %.4f Sam %.4f] [G %.4f] %s' %
                           (epoch, max_epochs,
                            batch_i, self.loader.n_batches,
                            di[0], di[1], di[2], di[3], ge[0], elapsed_time))

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
                    di = self.train_discriminator(x_train, clip)
                ge = self.train_generator()
    
                self.di_real_lss.append(di[1])
                self.di_fake_lss.append(di[2])
                self.di_lss.append(di[0])
                self.di_sample.append(di[3])
                self.ge_lss.append(ge[0])

                if epoch % show_every_n == 0:
                    elapsed_time = datetime.now() - s
                    print ('[Epochs %d/%d] [D Total %.4f Real %.4f Fake %.4f Sam %.4f] [G %.4f] Time %s' %
                           (epoch, max_epochs,
                            di[0], di[1], di[2], di[3], ge[0], elapsed_time))

                    self.show_img(self.generator, self.z_dim, file_name='sample_{}.png'.format(epoch), color='gray')
                    self.save_weights(file_name='weights_{}.h5'.format(epoch))
                    self.save_models(epoch=epoch)

            self.show_img(self.generator, self.z_dim, 'sample_last.png', color='gray')
            self.save_models()
            self.plot_loss()

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
