import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.nn import Network, Layer, get_trainable_params


class ConditionalGAN(object):
    def __init__(self,
                 condition_size, data_size,
                 generator_config, discriminator_config,
                 learning_rate=0.0001):
        self.generator_config = generator_config
        self.discriminator_config = discriminator_config
        self.condition = tf.placeholder(tf.float32, [None, condition_size])
        self.real_data = tf.placeholder(tf.float32, [None, data_size])
        self.learning_rate = learning_rate

        # generator
        self.generator = Network(self.condition, generator_config)
        self.generator_output = self.generator.get_output()

        # real discriminator
        self.discriminator_real = Network(
            tf.concat(1, [self.condition, self.real_data]), self.discriminator_config
        )
        self.discriminator_real_output = self.discriminator_real.get_output()

        # fake discriminator
        self.discriminator_fake = Network(
            tf.concat(1, [self.condition, self.generator_output]), self.discriminator_config
        )
        self.discriminator_fake_output = self.discriminator_fake.get_output()

        # discriminator objectives
        self.discriminator_cost = - tf.reduce_mean(
            tf.log(self.discriminator_real_output) +
            tf.log(1 - self.discriminator_fake_output)
        )
        self.discriminator_train = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.discriminator_cost,
            var_list=get_trainable_params(self.discriminator_config)
        )

        # generator objectives
        self.generator_cost = - tf.reduce_mean(
            tf.log(self.discriminator_fake_output)
        )
        self.generator_train = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.generator_cost,
            var_list=get_trainable_params(self.generator_config)
        )

        # extra scores
        self.rmse = tf.reduce_mean(tf.pow(self.real_data - self.generator_output, 2))

        # history losses
        self.generator_loss_history = []
        self.discriminator_loss_history = []
        self.rmse_history = []

    def train_generator(self, session, condition_and_data_sample):
        condition_sample, data_sample = condition_and_data_sample
        params = {
            self.condition: condition_sample,
            # self.real_data: data_sample,
        }
        session.run(self.generator_train, params)
        self.generator_loss_history.append(
            session.run(self.generator_cost, params)
        )

    def train_discriminator(self, session, condition_and_data_sample):
        condition_sample, data_sample = condition_and_data_sample
        params = {
            self.condition: condition_sample,
            self.real_data: data_sample,
        }
        session.run(self.discriminator_train, params)
        self.discriminator_loss_history.append(
            session.run(self.discriminator_cost, params)
        )

    def calculate_rmse(self, session, condition_and_data_sample):
        condition_sample, data_sample = condition_and_data_sample
        params = {
            self.condition: condition_sample,
            self.real_data: data_sample,
        }
        self.rmse_history.append(
            session.run(self.rmse, params)
        )

    def train(self, session, epochs, sample_condition_and_data, extra_discriminator_epochs=1, live=False):
        for epoch in xrange(epochs):

            for _ in xrange(extra_discriminator_epochs):
                self.train_discriminator(session, sample_condition_and_data())

            train_sample = sample_condition_and_data()
            self.train_generator(session, train_sample)
            self.calculate_rmse(session, train_sample)

            if live or epoch % 100 == 0:
                print 'epoch', epoch, 'd loss', self.discriminator_loss_history[-1],\
                    'g loss', self.generator_loss_history[-1], 'rmse', self.rmse_history[-1]


    def visualize_losses(self):
        plt.figure()
        plt.title('GAN losses')
        plt.plot(self.generator_loss_history, label='generator loss')
        plt.legend(loc='upper left')
        plt.show()
        plt.plot(self.discriminator_loss_history, label='discriminator loss')
        plt.legend(loc='upper left')
        plt.show()
        plt.plot(self.rmse_history, label='rmse')
        plt.legend(loc='upper left')
        plt.show()
