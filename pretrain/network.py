#!/usr/bin/env python3
"""Network model.

This module defines the network architecture used in the classification
training of the human demonstration data.

"""
import logging
import numpy as np
import tensorflow as tf

from abc import ABC
from abc import abstractmethod
from termcolor import colored

logger = logging.getLogger("network")


class Network(ABC):
    """Network base class."""

    use_mnih_2015 = False
    l1_beta = 0.
    l2_beta = 0.
    use_gpu = True

    def __init__(self, action_size, thread_index, device="/cpu:0"):
        """Initialize Network base class."""
        self.action_size = action_size
        self._thread_index = thread_index
        self._device = device

    @abstractmethod
    def prepare_loss(self):
        """Prepare tf operations training loss."""
        raise NotImplementedError()

    @abstractmethod
    def prepare_evaluate(self):
        """Prepare tf operations for evaluation."""
        raise NotImplementedError()

    @abstractmethod
    def load(self, sess, checkpoint):
        """Load existing model."""
        raise NotImplementedError()

    @abstractmethod
    def run_policy(self, sess, s_t):
        """Infer network output based on input s_t."""
        raise NotImplementedError()

    @abstractmethod
    def get_vars(self):
        """Return list of variables in the network."""
        raise NotImplementedError()

    def conv_variable(self, shape, layer_name='conv', gain=1.0, decoder=False):
        """Return weights and biases for convolutional 2D layer.

        Keyword arguments:
        shape -- [kernel_height, kernel_width, in_channel, out_channel]
        layer_name -- name of variables in the layer
        gain -- argument for orthogonal initializer (default 1.0)
        """
        with tf.variable_scope(layer_name):
            weight = tf.get_variable(
                'weights', shape,
                initializer=tf.orthogonal_initializer(gain=gain))
            bias = tf.get_variable(
                'biases', [shape[3] if not decoder else shape[2]],
                initializer=tf.zeros_initializer())
        return weight, bias

    def fc_variable(self, shape, layer_name='fc', gain=1.0):
        """Return weights and biases for dense layer.

        Keyword arguments:
        shape -- [# of units in, # of units out]
        layer_name -- name of variables in the layer
        gain -- argument for orthogonal initializer (default 1.0)
        """
        with tf.variable_scope(layer_name):
            weight = tf.get_variable(
                'weights', shape,
                initializer=tf.orthogonal_initializer(gain=gain))
            bias = tf.get_variable(
                'biases', [shape[1]], initializer=tf.zeros_initializer())
        return weight, bias

    def conv2d(self, x, W, stride, data_format='NHWC', padding="VALID",
               name=None):
        """Return convolutional 2d layer.

        Keyword arguments:
        x -- input
        W -- weights of layer with the shape
            [kernel_height, kernel_width, in_channel, out_channel]
        stride -- stride
        data_format -- NHWC or NCHW (default NHWC)
        padding -- SAME or VALID (default VALID)
        """
        return tf.nn.conv2d(
            x, W, strides=[1, stride, stride, 1], padding=padding,
            use_cudnn_on_gpu=self.use_gpu, data_format=data_format, name=name)

    def conv2d_transpose(self, x, W, output_shape, stride, data_format='NHWC',
                         padding='VALID', name=None):
        """Return transpose convolutional 2d layer.

        Keyword arguments:
        x -- input
        W -- weights of layer with the shapes
            [kernel_height, kernel_width, in_channel, out_channel]
        output_shape -- shape of the output
        stride -- stride
        data_format -- NHWC or NCHW (default NHWC)
        padding -- SAME or VALID (default VALID)
        """
        return tf.nn.conv2d_transpose(
            x, W, output_shape, strides=[1, stride, stride, 1],
            padding=padding, data_format=data_format, name=name)

    def build_grad_cam_grads(self):
        """Compute Grad-CAM from last convolutional layer after activation.

        Source:
        https://github.com/hiveml/tensorflow-grad-cam/blob/master/main.py
        """
        with tf.name_scope("GradCAM_Loss"):
            # We only care about target visualization class.
            signal = tf.multiply(self.logits, self.a)
            y_c = tf.reduce_sum(signal, axis=1)

            if self.use_mnih_2015:
                self.conv_layer = self.h_conv3
            else:
                self.conv_layer = self.h_conv2

            grads = tf.gradients(y_c, self.conv_layer)[0]
            # Normalizing the gradients
            self.grad_cam_grads = tf.div(
                grads, tf.sqrt(tf.reduce_mean(tf.square(grads)))
                + tf.constant(1e-5))

    def evaluate_grad_cam(self, sess, state, action):
        """Return activation and Grad-CAM of last convolutional layer.

        Keyword arguments:
        sess -- tf session
        state -- network input image
        action -- class label
        """
        activations, gradients = sess.run(
            [self.conv_layer, self.grad_cam_grads],
            feed_dict={self.s: [state], self.a: [action]})
        return activations[0], gradients[0]


class MultiClassNetwork(Network):
    """Multi-class Classification Network."""

    def __init__(self, action_size, thread_index, device="/cpu:0",
                 padding="VALID", in_shape=(84, 84, 4), use_sil=False):
        """Initialize MultiClassNetwork class."""
        Network.__init__(self, action_size, thread_index, device)
        self.graph = tf.Graph()
        logger.info("network: MultiClassNetwork")
        logger.info("action_size: {}".format(self.action_size))
        logger.info("use_mnih_2015: {}".format(
            colored(self.use_mnih_2015,
                    "green" if self.use_mnih_2015 else "red")))
        logger.info("L1_beta: {}".format(
            colored(self.l1_beta, "green" if self.l1_beta > 0. else "red")))
        logger.info("L2_beta: {}".format(
            colored(self.l2_beta, "green" if self.l2_beta > 0. else "red")))
        logger.info("padding: {}".format(padding))
        logger.info("in_shape: {}".format(in_shape))
        logger.info("use_sil: {}".format(
            colored(use_sil, "green" if use_sil else "red")))
        scope_name = "net_" + str(self._thread_index)
        self.last_hidden_fc_output_size = 512
        self.in_shape = in_shape
        self.use_sil = use_sil

        with self.graph.as_default():
            # state (input)
            self.s = tf.placeholder(tf.float32, [None] + list(self.in_shape))
            self.s_n = tf.div(self.s, 255.)

            with tf.device(self._device), tf.variable_scope(scope_name):
                if self.use_mnih_2015:
                    self.W_conv1, self.b_conv1 = self.conv_variable(
                        [8, 8, 4, 32], layer_name='conv1', gain=np.sqrt(2))
                    self.W_conv2, self.b_conv2 = self.conv_variable(
                        [4, 4, 32, 64], layer_name='conv2', gain=np.sqrt(2))
                    self.W_conv3, self.b_conv3 = self.conv_variable(
                        [3, 3, 64, 64], layer_name='conv3', gain=np.sqrt(2))

                    # 3136 for VALID padding and 7744 for SAME padding
                    fc1_size = 3136 if padding == 'VALID' else 7744
                    self.W_fc1, self.b_fc1 = self.fc_variable(
                        [fc1_size, self.last_hidden_fc_output_size],
                        layer_name='fc1', gain=np.sqrt(2))
                    tf.add_to_collection('transfer_params', self.W_conv1)
                    tf.add_to_collection('transfer_params', self.b_conv1)
                    tf.add_to_collection('transfer_params', self.W_conv2)
                    tf.add_to_collection('transfer_params', self.b_conv2)
                    tf.add_to_collection('transfer_params', self.W_conv3)
                    tf.add_to_collection('transfer_params', self.b_conv3)
                    tf.add_to_collection('transfer_params', self.W_fc1)
                    tf.add_to_collection('transfer_params', self.b_fc1)
                else:
                    logger.warn("Does not support SAME padding")
                    assert self.padding == 'VALID'
                    self.W_conv1, self.b_conv1 = self.conv_variable(
                        [8, 8, 4, 16], layer_name='conv1', gain=np.sqrt(2))
                    self.W_conv2, self.b_conv2 = self.conv_variable(
                        [4, 4, 16, 32], layer_name='conv2', gain=np.sqrt(2))
                    fc1_size = 2592
                    self.W_fc1, self.b_fc1 = self.fc_variable(
                        [fc1_size, self.last_hidden_fc_output_size],
                        layer_name='fc1', gain=np.sqrt(2))
                    tf.add_to_collection('transfer_params', self.W_conv1)
                    tf.add_to_collection('transfer_params', self.b_conv1)
                    tf.add_to_collection('transfer_params', self.W_conv2)
                    tf.add_to_collection('transfer_params', self.b_conv2)
                    tf.add_to_collection('transfer_params', self.W_fc1)
                    tf.add_to_collection('transfer_params', self.b_fc1)

                # weight for policy output layer
                self.W_fc2, self.b_fc2 = self.fc_variable(
                    [self.last_hidden_fc_output_size, action_size],
                    layer_name='fc2')
                tf.add_to_collection('transfer_params', self.W_fc2)
                tf.add_to_collection('transfer_params', self.b_fc2)

                if self.use_sil:
                    # weight for value output layer
                    self.W_fc3, self.b_fc3 = self.fc_variable(
                        [self.last_hidden_fc_output_size, 1], layer_name='fc3')
                    tf.add_to_collection('transfer_params', self.W_fc3)
                    tf.add_to_collection('transfer_params', self.b_fc3)

                if self.use_mnih_2015:
                    h_conv1 = tf.nn.relu(self.conv2d(
                        self.s_n,  self.W_conv1, 4, padding=padding)
                        + self.b_conv1)

                    h_conv2 = tf.nn.relu(self.conv2d(
                        h_conv1, self.W_conv2, 2, padding=padding)
                        + self.b_conv2)

                    self.h_conv3 = tf.nn.relu(self.conv2d(
                        h_conv2, self.W_conv3, 1, padding=padding)
                        + self.b_conv3)

                    h_conv3_flat = tf.reshape(self.h_conv3, [-1, fc1_size])
                    h_fc1 = tf.nn.relu(tf.matmul(
                        h_conv3_flat, self.W_fc1) + self.b_fc1)
                else:
                    h_conv1 = tf.nn.relu(self.conv2d(
                        self.s_n,  self.W_conv1, 4, padding=padding)
                        + self.b_conv1)

                    self.h_conv2 = tf.nn.relu(self.conv2d(
                        h_conv1, self.W_conv2, 2, padding=padding)
                        + self.b_conv2)

                    h_conv2_flat = tf.reshape(self.h_conv2, [-1, fc1_size])
                    h_fc1 = tf.nn.relu(tf.matmul(
                        h_conv2_flat, self.W_fc1) + self.b_fc1)

                # policy (output)
                self.logits = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2
                self.pi = tf.nn.softmax(self.logits)
                self.max_value = tf.reduce_max(self.logits, axis=None)

                if self.use_sil:
                    # value (output)
                    self.v = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
                    self.v0 = self.v[:, 0]

                self.saver = tf.train.Saver()

    def prepare_loss(self, sl_loss_weight=1.0, critic_weight=0.01):
        """Prepare tf operations training loss."""
        with self.graph.as_default():
            with tf.device(self._device), tf.name_scope("Loss"):
                # taken action (input for policy)
                self.a = tf.placeholder(tf.float32,
                                        shape=[None, self.action_size])

                sl_xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self.a, logits=self.logits)

                if self.use_sil:
                    self.returns = tf.placeholder(
                        tf.float32, shape=[None], name="sil_loss")

                    v_estimate = tf.squeeze(self.v)
                    advs = self.returns - v_estimate
                    clipped_advs = tf.maximum(advs, tf.zeros_like(advs))
                    sil_policy_loss = sl_xentropy \
                        * tf.stop_gradient(clipped_advs)

                    val_error = v_estimate - self.returns
                    clipped_val = tf.maximum(val_error,
                                             tf.zeros_like(val_error))
                    sil_val_loss = tf.square(clipped_val) * 0.5

                    self.sl_loss = tf.reduce_mean(sil_policy_loss) \
                        * sl_loss_weight
                    self.sl_loss += tf.reduce_mean(sil_val_loss) \
                        * critic_weight
                else:
                    self.sl_loss = tf.reduce_mean(sl_xentropy)

                self.total_loss = self.sl_loss

                net_vars = self.get_vars_no_bias()
                if self.l1_beta > 0:
                    l1_loss = tf.add_n(
                        [tf.reduce_sum(tf.abs(net_vars[i]))
                         for i in range(len(net_vars))]) * self.l1_beta
                    self.total_loss += l1_loss

                if self.l2_beta > 0:
                    l2_loss = tf.add_n(
                        [tf.nn.l2_loss(net_vars[i])
                         for i in range(len(net_vars))]) * self.l2_beta
                    self.total_loss += l2_loss

    def run_policy(self, sess, s_t):
        """Infer network output based on input s_t."""
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]

    def get_vars(self):
        """Return list of variables in the network."""
        if self.use_mnih_2015:
            vars = [
                self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_conv3, self.b_conv3,
                self.W_fc1, self.b_fc1,
                self.W_fc2, self.b_fc2,
                ]
        else:
            vars = [
                self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_fc1, self.b_fc1,
                self.W_fc2, self.b_fc2,
                ]

        if self.use_sil:
            vars.extend([self.W_fc3, self.b_fc3])

        return vars

    def get_vars_no_bias(self):
        """Return list of variables in the network excluding bias."""
        if self.use_mnih_2015:
            vars = [
                self.W_conv1, self.W_conv2,
                self.W_conv3, self.W_fc1, self.W_fc2,
                ]
        else:
            vars = [self.W_conv1, self.W_conv2, self.W_fc1, self.W_fc2]

        if self.use_sil:
            vars.extend([self.W_fc3])

        return vars

    def load(self, sess=None, checkpoint=''):
        """Load existing model."""
        assert sess is not None
        assert checkpoint != ''
        self.saver.restore(sess, checkpoint)
        logger.info("Successfully loaded: {}".format(checkpoint))

    def prepare_evaluate(self):
        """Prepare tf operations for evaluation."""
        with self.graph.as_default():
            with tf.device(self._device):
                correct_prediction = tf.equal(
                    tf.argmax(self.logits, 1), tf.argmax(self.a, 1))
                self.accuracy = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32))


# AutoEncoder-Classification Network
class AutoEncoderNetwork(Network):
    """AutoEncoder Network class."""

    def __init__(self, action_size, thread_index, device="/cpu:0",
                 padding="SAME", in_shape=(84, 84, 4), sae=False,
                 tied_weights=False, use_denoising=False, noise_factor=0.5,
                 loss_function='mse', use_sil=False):
        """Initialize AutoEncoderNetwork class."""
        Network.__init__(self, action_size, thread_index, device)
        assert self.use_mnih_2015
        self.graph = tf.Graph()
        logger.info("network: AutoEncoderNetwork")
        logger.info("action_size: {}".format(self.action_size))
        logger.info("use_mnih_2015: {}".format(
            colored(self.use_mnih_2015,
                    "green" if self.use_mnih_2015 else "red")))
        logger.info("L1_beta: {}".format(
            colored(self.l1_beta, "green" if self.l1_beta > 0. else "red")))
        logger.info("L2_beta: {}".format(
            colored(self.l2_beta, "green" if self.l2_beta > 0. else "red")))
        logger.info("padding: {}".format(padding))
        logger.info("use_denoising: {}".format(
            colored(use_denoising, "green" if use_denoising else "red")))
        if use_denoising:
            logger.info("noise_factor: {}".format(noise_factor))
        logger.info("loss_function: {}".format(loss_function))
        logger.info("in_shape: {}".format(in_shape))
        logger.info("use_sil: {}".format(
            colored(use_sil, "green" if use_sil else "red")))
        scope_name = "net_" + str(self._thread_index)
        self.last_hidden_fc_output_size = 512
        self.sae = sae  # supervised auto-encoder
        self.tied_weights = tied_weights
        self.use_denoising = use_denoising
        self.loss_function = loss_function
        self.in_shape = in_shape
        self.use_sil = use_sil

        with self.graph.as_default():
            # state (input)
            self.s = tf.placeholder(tf.float32, [None] + list(self.in_shape))
            self.s_norm = tf.div(self.s, 255.)

            # Denoising AE using dropout
            if self.use_denoising:
                self.training = tf.placeholder_with_default(
                    False, shape=(), name='training')
                self.s_norm_drop = tf.layers.dropout(
                    self.s_norm, noise_factor, training=self.training)

            assert self.in_shape[0] == self.in_shape[1]
            with tf.device(self._device), tf.variable_scope(scope_name):
                # Encoder
                self.W_conv1, self.b_conv1 = self.conv_variable(
                    [8, 8, 4, 32], layer_name='conv1', gain=np.sqrt(2))
                if padding == 'SAME':
                    shape_conv1 = np.ceil(self.in_shape[0] / 4.0)
                else:  # VALID
                    shape_conv1 = np.ceil(
                        (self.in_shape[0] - 8.0 + 1.0) / 4.0)

                self.W_conv2, self.b_conv2 = self.conv_variable(
                    [4, 4, 32, 64], layer_name='conv2', gain=np.sqrt(2))
                if padding == 'SAME':
                    shape_conv2 = np.ceil(shape_conv1 / 2.0)
                else:  # VALID
                    shape_conv2 = np.ceil(
                        (shape_conv1 - 4.0 + 1.0) / 2.0)

                self.W_conv3, self.b_conv3 = self.conv_variable(
                    [3, 3, 64, 64], layer_name='conv3', gain=np.sqrt(2))
                if padding == 'SAME':
                    shape_conv3 = int(np.ceil(shape_conv2 / 1.0))
                else:  # VALID
                    shape_conv3 = int(np.ceil(
                        (shape_conv2 - 3.0 + 1.0) / 1.0))

                # 3136 for VALID padding and 7744 for SAME padding
                conv3_output_size = shape_conv3 * shape_conv3 * 64
                self.W_fc1, self.b_fc1 = self.fc_variable(
                    [conv3_output_size, self.last_hidden_fc_output_size],
                    layer_name='fc1', gain=np.sqrt(2))
                tf.add_to_collection('transfer_params', self.W_conv1)
                tf.add_to_collection('transfer_params', self.b_conv1)
                tf.add_to_collection('transfer_params', self.W_conv2)
                tf.add_to_collection('transfer_params', self.b_conv2)
                tf.add_to_collection('transfer_params', self.W_conv3)
                tf.add_to_collection('transfer_params', self.b_conv3)
                tf.add_to_collection('transfer_params', self.W_fc1)
                tf.add_to_collection('transfer_params', self.b_fc1)

                # weight for policy output layer
                self.W_fc2, self.b_fc2 = self.fc_variable(
                    [self.last_hidden_fc_output_size, action_size],
                    layer_name='fc2')
                tf.add_to_collection('transfer_params', self.W_fc2)
                tf.add_to_collection('transfer_params', self.b_fc2)

                if self.use_sil:
                    # weight for value output layer
                    self.W_fc3, self.b_fc3 = self.fc_variable(
                        [self.last_hidden_fc_output_size, 1], layer_name='fc3')
                    tf.add_to_collection('transfer_params', self.W_fc3)
                    tf.add_to_collection('transfer_params', self.b_fc3)

                # Decoder
                self.d_W_fc1, self.d_b_fc1 = self.fc_variable(
                    [self.last_hidden_fc_output_size, conv3_output_size],
                    layer_name='d_fc1')
                self.d_W_conv3, self.d_b_conv3 = self.conv_variable(
                    [3, 3, 64, 64], layer_name='d_conv3', gain=np.sqrt(2),
                    decoder=True)
                self.d_W_conv2, self.d_b_conv2 = self.conv_variable(
                    [4, 4, 32, 64], layer_name='d_conv2', gain=np.sqrt(2),
                    decoder=True)
                self.d_W_conv1, self.d_b_conv1 = self.conv_variable(
                    [8, 8, 4, 32], layer_name='d_conv1', gain=np.sqrt(2),
                    decoder=True)

                input = self.s_norm_drop if self.use_denoising else self.s_norm
                # Encoder
                h_conv1 = tf.nn.relu(self.conv2d(
                    input,  self.W_conv1, 4, padding=padding)
                    + self.b_conv1)

                h_conv2 = tf.nn.relu(self.conv2d(
                    h_conv1, self.W_conv2, 2, padding=padding)
                    + self.b_conv2)

                self.h_conv3 = tf.nn.relu(self.conv2d(
                    h_conv2, self.W_conv3, 1, padding=padding)
                    + self.b_conv3)

                h_conv3_flat = tf.reshape(
                    self.h_conv3, [-1, conv3_output_size])
                h_fc1 = tf.nn.relu(tf.matmul(
                    h_conv3_flat, self.W_fc1) + self.b_fc1)

                # Classifier
                # policy (output)
                self.logits = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2
                self.pi = tf.nn.softmax(self.logits)
                self.max_value = tf.reduce_max(self.logits, axis=None)

                if self.use_sil:
                    # value (output)
                    self.v = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
                    self.v0 = self.v[:, 0]

                if self.tied_weights:
                    self.d_W_fc1 = tf.transpose(self.W_fc1)
                    self.d_W_conv3 = self.W_conv3
                    self.d_W_conv2 = self.W_conv2
                    self.d_W_conv1 = self.W_conv1

                # Decoder
                d_h_fc1 = tf.nn.relu(
                    tf.matmul(h_fc1, self.d_W_fc1) + self.d_b_fc1)

                d_h_fc1_wchannels = tf.reshape(
                    d_h_fc1, shape=[-1, shape_conv3, shape_conv3, 64])

                shape = tf.shape(h_conv2)
                out_shape = tf.stack(
                    [tf.shape(self.s)[0], shape[1], shape[2], shape[3]])
                d_h_conv3 = tf.nn.relu(self.conv2d_transpose(
                    d_h_fc1_wchannels, self.d_W_conv3, out_shape, 1,
                    padding=padding) + self.d_b_conv3)

                shape = tf.shape(h_conv1)
                out_shape = tf.stack(
                    [tf.shape(self.s)[0], shape[1], shape[2], shape[3]])
                d_h_conv2 = tf.nn.relu(self.conv2d_transpose(
                    d_h_conv3, self.d_W_conv2, out_shape, 2,
                    padding=padding) + self.d_b_conv2)

                out_shape = tf.stack([tf.shape(self.s)[0], self.in_shape[0],
                                      self.in_shape[1], self.in_shape[2]])
                self.decoder_out = self.conv2d_transpose(
                    d_h_conv2, self.d_W_conv1, out_shape, 4,
                    padding=padding) + self.d_b_conv1

                if self.loss_function == 'bce':
                    self.reconstruction = tf.nn.sigmoid(
                        self.decoder_out)
                else:
                    self.reconstruction = self.decoder_out

                self.saver = tf.train.Saver()

                logger.info("h_conv1 {}".format(h_conv1.get_shape()))
                logger.info("h_conv2 {}".format(h_conv2.get_shape()))
                logger.info("h_conv3 {}".format(self.h_conv3.get_shape()))
                logger.info("h_fc1 {}".format(h_fc1.get_shape()))
                logger.info("d_h_fc1 {}".format(d_h_fc1.get_shape()))
                logger.info("d_h_fc1_wchannels {}".format(
                    d_h_fc1_wchannels.get_shape()))
                logger.info("d_h_conv3 {}".format(d_h_conv3.get_shape()))
                logger.info("d_h_conv2 {}".format(d_h_conv2.get_shape()))
                logger.info("d_h_conv1 {}".format(
                    self.decoder_out.get_shape()))

    def prepare_loss(self, sl_loss_weight=1.0, critic_weight=0.005):
        """Prepare tf operations training loss."""
        with self.graph.as_default():
            with tf.device(self._device), tf.name_scope("Loss"):
                # taken action (input for policy)
                self.a = tf.placeholder(
                    tf.float32, shape=[None, self.action_size])

                sl_xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self.a, logits=self.logits)

                if self.use_sil:
                    self.returns = tf.placeholder(
                        tf.float32, shape=[None], name="sil_loss")

                    v_estimate = tf.squeeze(self.v)
                    advs = self.returns - v_estimate
                    clipped_advs = tf.maximum(advs, tf.zeros_like(advs))
                    sil_policy_loss = sl_xentropy \
                        * tf.stop_gradient(clipped_advs)

                    val_error = v_estimate - self.returns
                    clipped_val = tf.maximum(val_error,
                                             tf.zeros_like(val_error))
                    sil_val_loss = tf.square(clipped_val) * 0.5

                    self.sl_loss = tf.reduce_mean(sil_policy_loss) \
                        * sl_loss_weight
                    self.sl_loss += tf.reduce_mean(sil_val_loss) \
                        * critic_weight
                else:
                    self.sl_loss = tf.reduce_mean(sl_xentropy)

                if self.loss_function == 'bce':
                    # Autoencoder Loss (BCE)
                    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.s_norm, logits=self.decoder_out)
                    ae_error = tf.reduce_mean(xentropy, axis=(1, 2, 3))
                else:
                    # Autoencoder Loss (MSE)
                    # Squared difference of each batch element:
                    diff = tf.square(self.decoder_out - self.s_norm)
                    ae_error = 0.5 * tf.reduce_mean(diff, axis=(1, 2, 3))

                self.ae_loss = tf.reduce_mean(ae_error)
                if self.sae:  # supervised auto-encoder
                    if self.use_sil:
                        self.total_loss = tf.reduce_mean(
                            (sil_policy_loss * sl_loss_weight)
                            + (sil_val_loss * 0.005) + ae_error)
                    else:
                        self.total_loss = tf.reduce_mean(
                            (sl_xentropy * sl_loss_weight) + ae_error)
                else:
                    self.total_loss = self.sl_loss

                # L1/L2 regularization
                net_vars = self.get_vars_no_bias()
                ae_net_vars = self.get_vars_no_bias_ae_only()
                if self.l1_beta > 0:
                    l1_loss = tf.add_n(
                        [tf.reduce_sum(tf.abs(net_vars[i]))
                         for i in range(len(net_vars))]) * self.l1_beta
                    self.total_loss += l1_loss

                    if not self.sae:
                        ae_l1_loss = tf.add_n(
                            [tf.reduce_sum(tf.abs(ae_net_vars[i]))
                             for i in range(len(ae_net_vars))]) * self.l1_beta
                        self.ae_loss += ae_l1_loss

                if self.l2_beta > 0:
                    l2_loss = tf.add_n(
                        [tf.nn.l2_loss(net_vars[i])
                         for i in range(len(net_vars))]) * self.l2_beta
                    self.total_loss += l2_loss

                    if not self.sae:
                        ae_l2_loss = tf.add_n(
                            [tf.nn.l2_loss(ae_net_vars[i])
                             for i in range(len(ae_net_vars))]) * self.l2_beta
                        self.ae_loss += ae_l2_loss

    def run_policy(self, sess, s_t):
        """Infer network output based on input s_t."""
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]

    def get_vars(self):
        """Return list of variables in the network."""
        vars = None
        if self.sae:
            # Encoder vars
            vars = [
                self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_conv3, self.b_conv3,
                self.W_fc1, self.b_fc1,
                ]

            # Supervised vars
            vars.extend([self.W_fc2, self.b_fc2])

            if not self.tied_weights:
                # Decoder vars
                vars.extend([
                    self.d_W_conv1, self.d_b_conv1,
                    self.d_W_conv2, self.d_b_conv2,
                    self.d_W_conv3, self.d_b_conv3,
                    self.d_W_fc1, self.d_b_fc1,
                    ])
        else:
            # train only output layer for classifier when using AE
            vars = [self.W_fc2, self.b_fc2]

        if self.use_sil:
            vars.extend([self.W_fc3, self.b_fc3])

        return vars

    def get_vars_no_bias_ae_only(self):
        """Return list of variables in the network excluding bias."""
        vars = [self.W_conv1, self.W_conv2, self.W_conv3, self.W_fc1]

        # if not self.tied_weights:
        #     vars.extend([
        #         self.d_W_conv1, self.d_W_conv2,
        #         self.d_W_conv3, self.d_W_fc1,
        #         ])

        return vars

    def get_vars_no_bias(self):
        """Return list of variables in the network excluding bias."""
        vars = None
        if self.sae:
            vars = self.get_vars_no_bias_ae_only()
            vars.extend([self.W_fc2])  # Supervised var
        else:
            # train only output layer for classifier when using AE
            vars = [self.W_fc2]

        if self.use_sil:
            vars.extend([self.W_fc3])

        return vars

    def load(self, sess=None, checkpoint=''):
        """Load existing model."""
        assert sess is not None
        assert checkpoint != ''
        self.saver.restore(sess, checkpoint)
        logger.info("Successfully loaded: {}".format(checkpoint))

    def prepare_evaluate(self):
        """Prepare tf operations for evaluation."""
        with self.graph.as_default():
            with tf.device(self._device):
                correct_prediction = tf.equal(
                    tf.argmax(self.logits, 1), tf.argmax(self.a, 1))
                self.accuracy = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32))

    def reconstruct_image(self, sess, s_t):
        """Infer AE output based on input s_t."""
        image = sess.run(self.reconstruction, feed_dict={self.s: [s_t]})
        return image[0]


class MTLBinaryClassNetwork(Network):
    """MTL Binary Classification Network."""

    def __init__(self, action_size, thread_index, device="/cpu:0",
                 padding="VALID", in_shape=(84, 84, 4)):
        """Initialize MTLBinaryClassNetwork class."""
        Network.__init__(self, action_size, thread_index, device)
        self.graph = tf.Graph()
        logger.info("network: MTLBinaryClassNetwork")
        logger.info("action_size: {}".format(self.action_size))
        logger.info("use_mnih_2015: {}".format(
            colored(self.use_mnih_2015,
                    "green" if self.use_mnih_2015 else "red")))
        logger.info("L1_beta: {}".format(
            colored(self.l1_beta, "green" if self.l1_beta > 0. else "red")))
        logger.info("L2_beta: {}".format(
            colored(self.l2_beta, "green" if self.l2_beta > 0. else "red")))
        logger.info("padding: {}".format(padding))
        logger.info("in_shape: {}".format(in_shape))
        scope_name = "net_" + str(self._thread_index)
        self.last_hidden_fc_output_size = 512
        self.in_shape = in_shape

        with self.graph.as_default():
            # state (input)
            self.s = tf.placeholder(tf.float32, [None] + list(self.in_shape))
            self.s_n = tf.div(self.s, 255.)

            with tf.device(self._device), tf.variable_scope(scope_name):
                if self.use_mnih_2015:
                    self.W_conv1, self.b_conv1 = self.conv_variable(
                        [8, 8, 4, 32], layer_name='conv1', gain=np.sqrt(2))
                    self.W_conv2, self.b_conv2 = self.conv_variable(
                        [4, 4, 32, 64], layer_name='conv2', gain=np.sqrt(2))
                    self.W_conv3, self.b_conv3 = self.conv_variable(
                        [3, 3, 64, 64], layer_name='conv3', gain=np.sqrt(2))

                    # 3136 for VALID padding and 7744 for SAME padding
                    fc1_size = 3136 if padding == 'VALID' else 7744
                    self.W_fc1, self.b_fc1 = self.fc_variable(
                        [fc1_size, self.last_hidden_fc_output_size],
                        layer_name='fc1', gain=np.sqrt(2))
                    tf.add_to_collection('transfer_params', self.W_conv1)
                    tf.add_to_collection('transfer_params', self.b_conv1)
                    tf.add_to_collection('transfer_params', self.W_conv2)
                    tf.add_to_collection('transfer_params', self.b_conv2)
                    tf.add_to_collection('transfer_params', self.W_conv3)
                    tf.add_to_collection('transfer_params', self.b_conv3)
                    tf.add_to_collection('transfer_params', self.W_fc1)
                    tf.add_to_collection('transfer_params', self.b_fc1)
                else:
                    logger.warn("Does not support SAME padding")
                    assert padding == 'VALID'
                    self.W_conv1, self.b_conv1 = self.conv_variable(
                        [8, 8, 4, 16], layer_name='conv1', gain=np.sqrt(2))
                    self.W_conv2, self.b_conv2 = self.conv_variable(
                        [4, 4, 16, 32], layer_name='conv2', gain=np.sqrt(2))

                    fc1_size = 2592
                    self.W_fc1, self.b_fc1 = self.fc_variable(
                        [fc1_size, self.last_hidden_fc_output_size],
                        layer_name='fc1', gain=np.sqrt(2))
                    tf.add_to_collection('transfer_params', self.W_conv1)
                    tf.add_to_collection('transfer_params', self.b_conv1)
                    tf.add_to_collection('transfer_params', self.W_conv2)
                    tf.add_to_collection('transfer_params', self.b_conv2)
                    tf.add_to_collection('transfer_params', self.W_fc1)
                    tf.add_to_collection('transfer_params', self.b_fc1)

                # weight for policy output layer
                self.W_fc2, self.b_fc2 = [], []
                for n_class in range(action_size):
                    W, b = self.fc_variable(
                        [self.last_hidden_fc_output_size, 2],
                        layer_name='fc2_{}'.format(n_class))
                    self.W_fc2.append(W)
                    self.b_fc2.append(b)
                    tf.add_to_collection('transfer_params',
                                         self.W_fc2[n_class])
                    tf.add_to_collection('transfer_params',
                                         self.b_fc2[n_class])

                if self.use_mnih_2015:
                    h_conv1 = tf.nn.relu(
                        self.conv2d(self.s_n, self.W_conv1, 4, padding=padding)
                        + self.b_conv1)
                    h_conv2 = tf.nn.relu(
                        self.conv2d(h_conv1, self.W_conv2, 2, padding=padding)
                        + self.b_conv2)
                    self.h_conv3 = tf.nn.relu(
                        self.conv2d(h_conv2, self.W_conv3, 1, padding=padding)
                        + self.b_conv3)

                    h_conv3_flat = tf.reshape(self.h_conv3, [-1, fc1_size])
                    h_fc1 = tf.nn.relu(
                        tf.matmul(h_conv3_flat, self.W_fc1) + self.b_fc1)
                else:
                    h_conv1 = tf.nn.relu(
                        self.conv2d(self.s_n, self.W_conv1, 4, padding=padding)
                        + self.b_conv1)
                    self.h_conv2 = tf.nn.relu(
                        self.conv2d(h_conv1, self.W_conv2, 2, padding=padding)
                        + self.b_conv2)

                    fc1_size = 2592
                    h_conv2_flat = tf.reshape(self.h_conv2, [-1, fc1_size])
                    h_fc1 = tf.nn.relu(
                        tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

                # policy (output)
                self.logits, self.pi = [], []
                self.max_value = []
                for n_class in range(action_size):
                    logits = tf.add(
                        tf.matmul(h_fc1, self.W_fc2[n_class]),
                        self.b_fc2[n_class])
                    self.logits.append(logits)
                    pi = tf.nn.softmax(self.logits[n_class])
                    self.pi.append(pi)
                    max_value = tf.reduce_max(self.logits[n_class], axis=None)
                    self.max_value.append(max_value)

                self.saver = tf.train.Saver()

    def prepare_loss(self):
        """Prepare tf operations training loss."""
        with self.graph.as_default():
            with tf.device(self._device), tf.name_scope("Loss"):
                # taken action (input for policy)
                self.a = tf.placeholder(tf.float32, shape=[None, 2])
                self.reward = tf.placeholder(tf.float32, shape=[None, 1])

                if self.l1_beta > 0:
                    l1_regularizers = tf.reduce_sum(tf.abs(self.W_conv1)) \
                        + tf.reduce_sum(tf.abs(self.W_conv2)) \
                        + tf.reduce_sum(tf.abs(self.W_fc1))
                    if self.use_mnih_2015:
                        l1_regularizers += tf.reduce_sum(tf.abs(self.W_conv3))
                if self.l2_beta > 0:
                    l2_regularizers = tf.nn.l2_loss(self.W_conv1) \
                        + tf.nn.l2_loss(self.W_conv2) \
                        + tf.nn.l2_loss(self.W_fc1)
                    if self.use_mnih_2015:
                        l2_regularizers += tf.nn.l2_loss(self.W_conv3)

                self.total_loss = []
                for n_class in range(self.action_size):
                    logits = self.logits[n_class]

                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=self.a,
                        logits=logits)
                    total_loss = tf.reduce_mean(loss)

                    if self.l1_beta > 0:
                        l1_loss = self.l1_beta * (
                            l1_regularizers
                            + tf.reduce_sum(tf.abs(self.W_fc2[n_class])))
                        total_loss += l1_loss
                    if self.l2_beta > 0:
                        l2_loss = self.l2_beta * (
                            l2_regularizers
                            + tf.nn.l2_loss(self.W_fc2[n_class]))
                        total_loss += l2_loss
                    self.total_loss.append(total_loss)

    def run_policy(self, sess, s_t):
        """Infer network output based on input s_t."""
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out

    def get_vars(self):
        """Return list of variables in the network."""
        if self.use_mnih_2015:
            return [
                self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_conv3, self.b_conv3,
                self.W_fc1, self.b_fc1,
                self.W_fc2, self.b_fc2,
                ]
        else:
            return [
                self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_fc1, self.b_fc1,
                self.W_fc2, self.b_fc2,
                ]

    def load(self, sess=None, checkpoint=''):
        """Load existing model."""
        assert sess is not None
        assert checkpoint != ''
        self.saver.restore(sess, checkpoint)
        logger.info("Successfully loaded: {}".format(checkpoint))

    def prepare_evaluate(self):
        """Prepare tf operations for evaluation."""
        with self.graph.as_default():
            with tf.device(self._device):
                self.accuracy = []
                for n_class in range(self.action_size):
                    amax_class = tf.argmax(self.logits[n_class], 1)
                    amax_a = tf.argmax(self.a, 1)
                    correct_prediction = tf.equal(amax_class, amax_a)
                    pred = tf.cast(correct_prediction, tf.float32)
                    pred = tf.reduce_mean(pred)
                    self.accuracy.append(pred)
