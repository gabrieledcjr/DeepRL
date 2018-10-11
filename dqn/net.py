#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from math import sqrt


class Network(object):
    use_gpu = True

    def __init__(
        self, sess, height=84, width=84, phi_length=4, n_actions=1, name="network", gamma=0.99, copy_interval=4,
        optimizer='RMS', learning_rate=0.00025, epsilon=0.01, decay=0.95, momentum=0., l2_decay=0.0001, error_clip=1.0,
        slow=False, tau=0.01, verbose=False, folder='_networks', decay_learning_rate=False, device="/cpu:0"):
        self.name = name

    def conv_variable(self, shape, layer_name='conv'):
        initial = self.he_initializer(
            shape,
            fan_in=shape[2] * shape[0] * shape[1],
            fan_out=shape[3] * shape[0] * shape[1])
        with tf.variable_scope(layer_name):
            weight = tf.Variable(initial, name='weights')
            bias = tf.Variable(tf.zeros([shape[3]]), name='biases')
        return weight, bias

    def fc_variable(self, shape, layer_name='fc'):
        initial = self.he_initializer(shape, fan_in=shape[0], fan_out=shape[1])
        with tf.variable_scope(layer_name):
            weight = tf.Variable(initial, name='weights')
            bias = tf.Variable(tf.zeros([shape[1]]), name='biases')
        return weight, bias

    def he_initializer(self, shape, fan_in=1.0, fan_out=1.0):
        return self.variance_scaling_initializer(
            shape, fan_in, fan_out, factor=2.0, mode='FAN_IN', uniform=False)

    def xavier_initializer(self, shape, fan_in, fan_out):
        return self.variance_scaling_initializer(
            shape, fan_in, fan_out, factor=1.0, mode='FAN_AVG', uniform=True)

    def variance_scaling_initializer(self, shape, fan_in, fan_out, factor=2.0, mode='FAN_IN', uniform=False):
        if mode == 'FAN_IN':
            n = fan_in
        elif mode == 'FAN_AVG':
            n = (fan_in + fan_out) / 2.0

        if uniform:
            limit = np.sqrt(3.0 * factor / n)
            # sampling from a uniform distribution
            return tf.random_uniform(shape, minval=-limit, maxval=limit, dtype=tf.float32)
        else:
            trunc_stddev = np.sqrt(1.3 * factor / n)
            return tf.truncated_normal(shape, mean=0.0, stddev=trunc_stddev, dtype=tf.float32)

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
