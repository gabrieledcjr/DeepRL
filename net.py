#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from math import sqrt


class Network(object):

    def __init__(
        self, sess, height=84, width=84, phi_length=4, n_actions=1, name="network", gamma=0.99, copy_interval=4,
        optimizer='RMS', learning_rate=0.00025, epsilon=0.01, decay=0.95, momentum=0., l2_decay=0.0001, error_clip=1.0,
        slow=False, tau=0.01, verbose=False, path='', folder='_networks', decay_learning_rate=False):
        self.name = name

    def weight_variable_(self, shape, layer_name, stddev=0.01):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial, name=self.name + '_' + layer_name + '_weights')

    def bias_variable_(self, shape, layer_name, value=0.01):
        initial = tf.constant(value, shape=[shape[-1]])
        return tf.Variable(initial, name=self.name + '_' + layer_name + '_biases')

    def weight_variable(self, shape, layer_name):
        fan_in = np.prod(shape[0:-1])
        std = 1. / sqrt(fan_in)
        initial = tf.random_uniform(shape, minval=(-std), maxval=std)
        return tf.Variable(initial, name=self.name + '_' + layer_name + '_weights')

    def bias_variable(self, shape, layer_name):
        """ Pass the same shape as was passed in the weight_variable  """
        fan_in = np.prod(shape[0:-1])
        std = 1. / sqrt(fan_in)
        initial = tf.random_uniform([shape[-1]], minval=(-std), maxval=std)
        return tf.Variable(initial, name=self.name + '_' + layer_name + '_biases')

    def weight_variable_last_layer(self, shape, layer_name):
        std = 0.003
        initial = tf.random_uniform(shape, minval=(-std), maxval=std)
        return tf.Variable(initial, name=self.name + '_' + layer_name + '_weights')

    def bias_variable_last_layer(self, shape, layer_name):
        std = 0.003
        initial = tf.random_uniform([shape[-1]], minval=(-std), maxval=std)
        return tf.Variable(initial, name=self.name + '_' + layer_name + '_biases')

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding = "VALID")

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
