# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import logging

from abc import ABC, abstractmethod
from time import sleep
from termcolor import colored

logger = logging.getLogger("game_ac_network")

# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(ABC):
    use_mnih_2015 = False
    l1_beta = 0.
    l2_beta = 0.
    use_gpu = False

    def __init__(self,
                 action_size,
                 thread_index, # -1 for global
                 device="/cpu:0"):
        self._action_size = action_size
        self._thread_index = thread_index
        self._device = device

    def prepare_loss(self):
        with tf.device(self._device):
            # taken action (input for policy)
            self.a = tf.placeholder(tf.float32, [None, self._action_size])

            # temporary difference (R-V) (input for policy)
            self.td = tf.placeholder(tf.float32, [None])

            # avoid NaN with clipping when value in pi becomes zero
            #log_pi = tf.nn.log_softmax(tf.clip_by_value(self.logits, -10.0, 10.0))
            log_pi = tf.nn.log_softmax(self.logits)

            # policy entropy
            entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)

            # net_vars = self.get_vars()
            # l2_losses = []
            # l1_losses = []
            # for i in range(len(net_vars)):
            # if i%2 == 0:
            #       l1_losses.append(self.l1_beta * tf.reduce_sum(tf.abs(net_vars[i])))
            #       l2_losses.append(self.l2_beta * tf.nn.l2_loss(net_vars[i]))
            # l1_loss = sum(l1_losses)
            # l2_loss = sum(l2_losses)
            # l_losses = l1_loss + l2_loss

            self.policy_lr = tf.placeholder(tf.float32, shape=(), name="policy_lr")
            self.critic_lr = tf.placeholder(tf.float32, shape=(), name="critic_lr")
            self.entropy_beta = tf.placeholder(tf.float32, shape=(), name="entropy_beta")

            # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
            policy_loss = -tf.reduce_sum( tf.reduce_sum( tf.multiply(log_pi, self.a), axis=1 ) * self.td + entropy * self.entropy_beta)
            # policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.multiply( log_pi, self.a ) + l_losses, axis=1 ) * self.td + entropy * entropy_beta)

            # R (input for value)
            self.r = tf.placeholder(tf.float32, [None])

            # value loss (output)
            # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
            value_loss = 0.5 * tf.reduce_sum(tf.square(self.r - self.v))

            # gradient of policy and value are summed up
            self.total_loss = self.policy_lr * policy_loss + self.critic_lr * value_loss

    @abstractmethod
    def run_policy_and_value(self, sess, s_t):
        raise NotImplementedError()

    @abstractmethod
    def run_policy(self, sess, s_t):
        raise NotImplementedError()

    @abstractmethod
    def run_value(self, sess, s_t):
        raise NotImplementedError()

    @abstractmethod
    def get_vars(self):
        raise NotImplementedError()

    def sync_from(self, src_network, name=None, upper_layers_only=False):
        if upper_layers_only:
            src_vars = src_network.get_vars_upper()
            dst_vars = self.get_vars_upper()
        else:
            src_vars = src_network.get_vars()
            dst_vars = self.get_vars()

        sync_ops = []

        with tf.device(self._device):
            with tf.name_scope(name, "GameACNetwork", []) as name:
                for(src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)

    def conv_variable(self, shape, layer_name='conv'):
        initial = self.xavier_initializer(
            shape,
            fan_in=shape[2] * shape[0] * shape[1],
            fan_out=shape[3] * shape[0] * shape[1])
        with tf.variable_scope(layer_name):
            weight = tf.Variable(initial, name='weights')
            bias = tf.Variable(tf.zeros([shape[3]]), name='biases')
        return weight, bias

    def fc_variable(self, shape, layer_name='fc'):
        initial = self.xavier_initializer(shape, fan_in=shape[0], fan_out=shape[1])
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

    def load_transfer_model(
        self, sess, folder='',
        not_transfer_fc2=False, not_transfer_fc1=False,
        not_transfer_conv3=False, not_transfer_conv2=False,
        var_list=None):
        assert folder != ''
        assert os.path.isdir(folder)
        assert self._thread_index == -1 # only load model to global network

        transfer_all = False
        if not_transfer_conv2:
            folder += '/noconv2'
        elif not_transfer_conv3:
            folder += '/noconv3'
        elif not_transfer_fc1:
            folder += '/nofc1'
        elif not_transfer_fc2:
            folder += '/nofc2'
        else:
            transfer_all = True
            with open(folder + "/max_output_value", 'r') as f_max_value:
                transfer_max_output_val = float(f_max_value.readline().split()[0])
            folder += '/all'

        saver_transfer_from = tf.train.Saver(var_list=var_list)
        checkpoint_transfer_from = tf.train.get_checkpoint_state(folder)

        if checkpoint_transfer_from and checkpoint_transfer_from.model_checkpoint_path:
            saver_transfer_from.restore(sess, checkpoint_transfer_from.model_checkpoint_path)
            logger.info("Successfully loaded: {}".format(checkpoint_transfer_from.model_checkpoint_path))

            global_vars = tf.global_variables()
            is_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            initialized_vars = [v for (v, f) in zip(global_vars, is_initialized) if f]
            for var in initialized_vars:
                logger.info("    {} loaded".format(var.op.name))
                sleep(.2)

            if transfer_all:
                # scale down last layer if it's transferred
                logger.info("Normalizing output layer with max value {}...".format(transfer_max_output_val))
                W_fc2_norm = tf.div(self.W_fc2, transfer_max_output_val)
                b_fc2_norm = tf.div(self.b_fc2, transfer_max_output_val)
                logger.info("Output layer normalized")
                sess.run([
                    self.W_fc2.assign(W_fc2_norm), self.b_fc2.assign(b_fc2_norm)
                ])

# Actor-Critic FF Network
class GameACFFNetwork(GameACNetwork):
    def __init__(self,
                 action_size,
                 thread_index, # -1 for global
                 device="/cpu:0"):
        GameACNetwork.__init__(self, action_size, thread_index, device)
        logger.info("use_mnih_2015: {}".format(colored(self.use_mnih_2015, "green" if self.use_mnih_2015 else "red")))
        scope_name = "net_" + str(self._thread_index)
        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            if self.use_mnih_2015:
                self.W_conv1, self.b_conv1 = self.conv_variable([8, 8, 4, 32], layer_name='conv1')
                self.W_conv2, self.b_conv2 = self.conv_variable([4, 4, 32, 64], layer_name='conv2')
                self.W_conv3, self.b_conv3 = self.conv_variable([3, 3, 64, 64], layer_name='conv3')
                self.W_fc1, self.b_fc1 = self.fc_variable([3136, 512], layer_name='fc1')
            else:
                self.W_conv1, self.b_conv1 = self.conv_variable([8, 8, 4, 16], layer_name='conv1')  # stride=4
                self.W_conv2, self.b_conv2 = self.conv_variable([4, 4, 16, 32], layer_name='conv2') # stride=2
                self.W_fc1, self.b_fc1 = self.fc_variable([2592, 512], layer_name='fc1')

            # weight for policy output layer
            self.W_fc2, self.b_fc2 = self.fc_variable([512, action_size], layer_name='fc2')

            # weight for value output layer
            self.W_fc3, self.b_fc3 = self.fc_variable([512, 1], layer_name='fc3')

            # state (input)
            self.s = tf.placeholder(tf.float32, [None, 84, 84, 4])
            self.s_n = tf.div(self.s, 255.)

            if self.use_mnih_2015:
                self.h_conv1 = tf.nn.relu(self.conv2d(self.s_n,  self.W_conv1, 4) + self.b_conv1)
                self.h_conv2 = tf.nn.relu(self.conv2d(self.h_conv1, self.W_conv2, 2) + self.b_conv2)
                self.h_conv3 = tf.nn.relu(self.conv2d(self.h_conv2, self.W_conv3, 1) + self.b_conv3)

                self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 3136])
                self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)
            else:
                self.h_conv1 = tf.nn.relu(self.conv2d(self.s_n,  self.W_conv1, 4) + self.b_conv1)
                self.h_conv2 = tf.nn.relu(self.conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

                h_conv2_flat = tf.reshape(self.h_conv2, [-1, 2592])
                self.h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

            # policy (output)
            self.logits = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2
            self.pi = tf.nn.softmax(self.logits)
            # value (output)
            v_ = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
            self.v = tf.reshape( v_, [-1] )

    def run_policy_and_value(self, sess, s_t):
        pi_out, v_out, logits = sess.run( [self.pi, self.v, self.logits], feed_dict = {self.s : [s_t]} )
        return (pi_out[0], v_out[0], logits[0])

    def run_policy(self, sess, s_t):
        pi_out = sess.run( self.pi, feed_dict = {self.s : [s_t]} )
        return pi_out[0]

    def run_value(self, sess, s_t):
        v_out = sess.run( self.v, feed_dict = {self.s : [s_t]} )
        return v_out[0]

    def get_vars(self):
        if self.use_mnih_2015:
            return [self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_conv3, self.b_conv3,
                self.W_fc1, self.b_fc1,
                self.W_fc2, self.b_fc2,
                self.W_fc3, self.b_fc3]
        else:
            return [self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_fc1, self.b_fc1,
                self.W_fc2, self.b_fc2,
                self.W_fc3, self.b_fc3]

    def get_vars_upper(self):
        return [self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3]

# Actor-Critic LSTM Network
class GameACLSTMNetwork(GameACNetwork):
    def __init__(self,
                 action_size,
                 thread_index, # -1 for global
                 device="/cpu:0"):
        GameACNetwork.__init__(self, action_size, thread_index, device)
        logger.info("use_mnih_2015: {}".format(colored(self.use_mnih_2015, "green" if self.use_mnih_2015 else "red")))
        scope_name = "net_" + str(self._thread_index)
        self.lstm_cells_size = 256
        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            if self.use_mnih_2015:
                self.W_conv1, self.b_conv1 = self.conv_variable([8, 8, 4, 32], layer_name='conv1')
                self.W_conv2, self.b_conv2 = self.conv_variable([4, 4, 32, 64], layer_name='conv2')
                self.W_conv3, self.b_conv3 = self.conv_variable([3, 3, 64, 64], layer_name='conv3')
                self.W_fc1, self.b_fc1 = self.fc_variable([3136, self.lstm_cells_size], layer_name='fc1')
            else:
                self.W_conv1, self.b_conv1 = self.conv_variable([8, 8, 4, 16], layer_name='conv1')  # stride=4
                self.W_conv2, self.b_conv2 = self.conv_variable([4, 4, 16, 32], layer_name='conv2') # stride=2
                self.W_fc1, self.b_fc1 = self.fc_variable([2592, self.lstm_cells_size], layer_name='fc1')

            # lstm
            self.lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_cells_size, name='basic_lstm_cell')

            # weight for policy output layer
            self.W_fc2, self.b_fc2 = self.fc_variable([self.lstm_cells_size, action_size], layer_name='fc2')

            # weight for value output layer
            self.W_fc3, self.b_fc3 = self.fc_variable([self.lstm_cells_size, 1], layer_name='fc3')

            # state (input)
            self.s = tf.placeholder(tf.float32, [None, 84, 84, 4])
            self.s_n = tf.div(self.s, 255.)

            if self.use_mnih_2015:
                self.h_conv1 = tf.nn.relu(self.conv2d(self.s_n,  self.W_conv1, 4) + self.b_conv1)
                self.h_conv2 = tf.nn.relu(self.conv2d(self.h_conv1, self.W_conv2, 2) + self.b_conv2)
                self.h_conv3 = tf.nn.relu(self.conv2d(self.h_conv2, self.W_conv3, 1) + self.b_conv3)

                h_conv3_flat = tf.reshape(self.h_conv3, [-1, 3136])
                self.h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.W_fc1) + self.b_fc1)
            else:
                self.h_conv1 = tf.nn.relu(self.conv2d(self.s_n, self.W_conv1, 4) + self.b_conv1)
                self.h_conv2 = tf.nn.relu(self.conv2d(self.h_conv1, self.W_conv2, 2) + self.b_conv2)

                h_conv2_flat = tf.reshape(self.h_conv2, [-1, 2592])
                self.h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

            h_fc1_reshaped = tf.reshape(self.h_fc1, [1, -1, self.lstm_cells_size])

            # place holder for LSTM unrolling time step size.
            self.step_size = tf.placeholder(tf.float32, [1])

            self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, self.lstm_cells_size])
            self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, self.lstm_cells_size])
            self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(
                self.initial_lstm_state0,
                self.initial_lstm_state1)

            # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
            # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
            # Unrolling step size is applied via self.step_size placeholder.
            # When forward propagating, step_size is 1.
            # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
            lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
                self.lstm,
                h_fc1_reshaped,
                initial_state = self.initial_lstm_state,
                sequence_length = self.step_size,
                time_major = False,
                scope = scope)

            # lstm_outputs: (1,5,256) for back prop, (1,1,256) for forward prop.

            lstm_outputs = tf.reshape(lstm_outputs, [-1, self.lstm_cells_size])

            # policy (output)
            self.logits = tf.matmul(lstm_outputs, self.W_fc2) + self.b_fc2
            self.pi = tf.nn.softmax(self.logits)

            # value (output)
            v_ = tf.matmul(lstm_outputs, self.W_fc3) + self.b_fc3
            self.v = tf.reshape(v_, [-1])

            scope.reuse_variables()
            self.W_lstm = tf.get_variable("basic_lstm_cell/kernel")
            self.b_lstm = tf.get_variable("basic_lstm_cell/bias")

            self.reset_state()

    def reset_state(self):
        self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(
            np.zeros([1, self.lstm_cells_size]),
            np.zeros([1, self.lstm_cells_size]))

    def run_policy_and_value(self, sess, s_t):
        # This run_policy_and_value() is used when forward propagating.
        # so the step size is 1.
        pi_out, v_out, self.lstm_state_out, logits = sess.run(
            [self.pi, self.v, self.lstm_state, self.logits],
            feed_dict = {
                self.s : [s_t],
                self.initial_lstm_state0 : self.lstm_state_out[0],
                self.initial_lstm_state1 : self.lstm_state_out[1],
                self.step_size : [1]})
        # pi_out: (1,3), v_out: (1)
        return (pi_out[0], v_out[0], logits[0])

    def run_policy(self, sess, s_t):
        # This run_policy() is used for displaying the result with display tool.
        pi_out, self.lstm_state_out = sess.run(
            [self.pi, self.lstm_state],
            feed_dict = {
                self.s : [s_t],
                self.initial_lstm_state0 : self.lstm_state_out[0],
                self.initial_lstm_state1 : self.lstm_state_out[1],
                self.step_size : [1]})

        return pi_out[0]

    def run_value(self, sess, s_t):
        # This run_value() is used for calculating V for bootstrapping at the
        # end of LOCAL_T_MAX time step sequence.
        # When next sequcen starts, V will be calculated again with the same state using updated network weights,
        # so we don't update LSTM state here.
        prev_lstm_state_out = self.lstm_state_out
        v_out, _ = sess.run(
            [self.v, self.lstm_state],
            feed_dict = {
                self.s : [s_t],
                self.initial_lstm_state0 : self.lstm_state_out[0],
                self.initial_lstm_state1 : self.lstm_state_out[1],
                self.step_size : [1]})

        # roll back lstm state
        self.lstm_state_out = prev_lstm_state_out
        return v_out[0]

    def get_vars(self):
        if self.use_mnih_2015:
            return [self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_conv3, self.b_conv3,
                self.W_fc1, self.b_fc1,
                self.W_lstm, self.b_lstm,
                self.W_fc2, self.b_fc2,
                self.W_fc3, self.b_fc3]
        else:
            return [self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_fc1, self.b_fc1,
                self.W_lstm, self.b_lstm,
                self.W_fc2, self.b_fc2,
                self.W_fc3, self.b_fc3]

    def get_vars_upper(self):
        return [self.W_fc1, self.b_fc1,
            self.W_lstm, self.b_lstm,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3]
