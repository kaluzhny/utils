import numpy as np
import tensorflow as tf


class Layer(object):
    def __init__(self, input_size, output_size, activation_function=tf.nn.relu,
                 weight_distribution=tf.random_normal):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.weight_distribution = weight_distribution
        self.initialize()

    def initialize(self):
        self.W = tf.Variable(self.weight_distribution([self.input_size, self.output_size]))
        self.b = tf.Variable(tf.zeros([self.output_size]))
        # self.b = tf.Variable(self.weight_distribution([self.output_size]))

    def apply(self, data):
        return self.activation_function(
            tf.matmul(data, self.W) + self.b
        )

    def get_params(self):
        return [self.W, self.b]


class DummyLayer(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def apply(self, data):
        return data

    def get_params(self):
        return []


class ConvolutionalLayer(object):
    # conv2d: https://www.tensorflow.org/api_docs/python/nn/convolution#conv2d
    # max_pool: https://www.tensorflow.org/api_docs/python/nn/pooling#max_pool
    def __init__(self, in_width, in_height, in_channels,
                 filter_width, filter_height, out_channels,
                 stride=1, max_pool_size=2, max_pool_stride=2,
                 activation_function=tf.nn.relu):
        self.in_width = in_width
        self.in_height = in_height
        self.in_channels = in_channels
        self.input_size = in_width * in_height * in_channels

        self.stride = stride
        self.max_pool_size= max_pool_size
        self.max_pool_stride = max_pool_stride
        self.activation_function = activation_function

        filter_shape = [filter_width, filter_height, in_channels, out_channels]
        self.W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1))
        self.b = tf.Variable(tf.constant(0.1, shape=[out_channels]))

    def apply(self, data):
        image = tf.reshape(data, [-1, self.in_height, self.in_width, self.in_channels])
        conv_hidden = tf.nn.conv2d(image, self.W, strides=[1, self.stride, self.stride, 1], padding='SAME')
        conv_hidden = self.activation_function(conv_hidden + self.b)
        pool_hidden = tf.nn.max_pool(conv_hidden,
                              ksize=[1, self.max_pool_size, self.max_pool_size, 1],
                              strides=[1, self.max_pool_stride, self.max_pool_stride, 1],
                              padding='SAME')
        shape = pool_hidden.get_shape()
        return tf.reshape(pool_hidden, [-1, int(shape[1]*shape[2]*shape[3])])

    def get_params(self):
        return [self.W, self.b]


class Network(object):
    def __init__(self, input_value, layers):
        self.input_value = input_value
        self.layers = layers

    def get_output(self):
        # apply layers one by one
        result = self.input_value
        for layer in self.layers:
            if isinstance(layer, list):
                subresults = []
                data_idx = 0
                for sublayer in layer:
                    subresults.append(sublayer.apply(result[:,data_idx:data_idx+sublayer.input_size]))
                    data_idx += sublayer.input_size
                result = tf.concat(1, subresults)
            else:
                result = layer.apply(result)
        return result


def get_trainable_params(layers):
    params = []
    for layer in layers:
        if isinstance(layer, list):
            for sublayer in layer:
                params.extend(sublayer.get_params())
        else:
            params.extend(layer.get_params())
    return params
