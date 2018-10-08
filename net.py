#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from math import sqrt


class Network(object):
    use_gpu = True

    def __init__(
        self, sess, height=84, width=84, phi_length=4, n_actions=1, name="network", gamma=0.99, copy_interval=4,
        optimizer='RMS', learning_rate=0.00025, epsilon=0.01, decay=0.95, momentum=0., l2_decay=0.0001, error_clip=1.0,
        slow=False, tau=0.01, verbose=False, path='', folder='_networks', decay_learning_rate=False):
        self.name = name

    def conv_variable(self, shape, layer_name=''):
        std = self._xavier_std(
            shape[2] * shape[0] * shape[1], # fan in
            shape[3] * shape[0] * shape[1]  # fan out
        )
        initial = tf.random_uniform(shape, minval=-std, maxval=std)
        weight = tf.Variable(initial, name=self.name + '_' + layer_name + '_weights')
        bias = tf.Variable(tf.zeros([shape[3]]), name=self.name + '_' + layer_name + '_biases')
        return weight, bias

    def fc_variable(self, shape, layer_name=''):
        std = self._xavier_std(shape[0], shape[1])
        initial = tf.random_uniform(shape, minval=-std, maxval=std)
        weight = tf.Variable(initial, name=self.name + '_' + layer_name + '_weights')
        bias = tf.Variable(tf.zeros([shape[1]]), name=self.name + '_' + layer_name + '_biases')
        return weight, bias

    def _xavier_std(self, fan_in, fan_out):
        # sampling from a uniform distribution
        return np.sqrt(6. / (fan_in + fan_out))

    def conv2d(self, x, W, stride, data_format='NHWC'):
        return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding = "VALID",
            use_cudnn_on_gpu=self.use_gpu, data_format=data_format)

    def variable_summaries(self, var, name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
            tf.summary.histogram(name, var)
