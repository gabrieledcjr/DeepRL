#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import logging

from abc import ABC, abstractmethod
from time import sleep
from termcolor import colored

logger = logging.getLogger("game_ac_network")


class GameACNetwork(ABC):
    """Actor-Critic Network Base Class
    (Policy network and Value network)
    """
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

    def prepare_loss(self, entropy_beta=0.01, critic_lr=0.5):
        """Based from A2C OpenAI Baselines
        A2C (aka PAAC), we use the loss function explained in the PAAC paper
        Clemente et al 2017. Efficient Parallel Methods for Deep Reinforcement Learning
        """

        def cat_entropy(logits):
            a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, 1, keepdims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

        with tf.name_scope("Loss") as scope:
            # taken action (input for policy)
            self.a = tf.placeholder(tf.float32, shape=[None, self._action_size], name="action")

            # temporal difference (R-V) (input for policy)
            self.advantage = tf.placeholder(tf.float32, shape=[None], name="advantage")
            self.cumulative_reward = tf.placeholder(tf.float32, shape=[None], name="cumulative_reward")

            assert self.a.shape.as_list() == self.logits.shape.as_list()
            neglogpac = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.a)
            pg_loss = tf.reduce_mean(self.advantage * neglogpac)
            #vf_loss = tf.losses.mean_squared_error(tf.squeeze(self.v), self.cumulative_reward)
            vf_loss = tf.reduce_mean(tf.squared_difference(tf.squeeze(self.v), self.cumulative_reward) / 2.0)
            entropy = tf.reduce_mean(cat_entropy(self.logits))
            self.total_loss = pg_loss - entropy * entropy_beta + vf_loss * critic_lr

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

    def conv_variable(self, shape, layer_name='conv', gain=1.0):
        with tf.variable_scope(layer_name):
            weight = tf.get_variable('weights', shape, initializer=tf.orthogonal_initializer(gain=gain))
            bias = tf.get_variable('biases', [shape[3]], initializer=tf.zeros_initializer())
        return weight, bias

    def fc_variable(self, shape, layer_name='fc', gain=1.0):
        with tf.variable_scope(layer_name):
            weight = tf.get_variable('weights', shape, initializer=tf.orthogonal_initializer(gain=gain))
            bias = tf.get_variable('biases', [shape[1]], initializer=tf.zeros_initializer())
        return weight, bias

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
        self.last_hidden_fc_output_size = 512

        # state (input)
        self.s = tf.placeholder(tf.float32, [None, 84, 84, 4], name="state")
        self.s_n = tf.div(self.s, 255.)

        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            if self.use_mnih_2015:
                self.W_conv1, self.b_conv1 = self.conv_variable([8, 8, 4, 32], layer_name='conv1', gain=np.sqrt(2))
                self.W_conv2, self.b_conv2 = self.conv_variable([4, 4, 32, 64], layer_name='conv2', gain=np.sqrt(2))
                self.W_conv3, self.b_conv3 = self.conv_variable([3, 3, 64, 64], layer_name='conv3', gain=np.sqrt(2))
                self.W_fc1, self.b_fc1 = self.fc_variable([3136, self.last_hidden_fc_output_size], layer_name='fc1', gain=np.sqrt(2))
            else:
                self.W_conv1, self.b_conv1 = self.conv_variable([8, 8, 4, 16], layer_name='conv1', gain=np.sqrt(2))  # stride=4
                self.W_conv2, self.b_conv2 = self.conv_variable([4, 4, 16, 32], layer_name='conv2', gain=np.sqrt(2)) # stride=2
                self.W_fc1, self.b_fc1 = self.fc_variable([2592, self.last_hidden_fc_output_size], layer_name='fc1', gain=np.sqrt(2))

            # weight for policy output layer
            self.W_fc2, self.b_fc2 = self.fc_variable([self.last_hidden_fc_output_size, action_size], layer_name='fc2')

            # weight for value output layer
            self.W_fc3, self.b_fc3 = self.fc_variable([self.last_hidden_fc_output_size, 1], layer_name='fc3')

            if self.use_mnih_2015:
                self.h_conv1 = tf.nn.relu(self.conv2d(self.s_n,  self.W_conv1, 4) + self.b_conv1)
                self.h_conv2 = tf.nn.relu(self.conv2d(self.h_conv1, self.W_conv2, 2) + self.b_conv2)
                self.h_conv3 = tf.nn.relu(self.conv2d(self.h_conv2, self.W_conv3, 1) + self.b_conv3)

                self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 3136])
                self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)
            else:
                self.h_conv1 = tf.nn.relu(self.conv2d(self.s_n,  self.W_conv1, 4) + self.b_conv1)
                self.h_conv2 = tf.nn.relu(self.conv2d(self.h_conv1, self.W_conv2, 2) + self.b_conv2)

                h_conv2_flat = tf.reshape(self.h_conv2, [-1, 2592])
                self.h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

            # policy (output)
            self.logits = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
            self.pi = tf.nn.softmax(self.logits)
            # value (output)
            self.v = tf.matmul(self.h_fc1, self.W_fc3) + self.b_fc3
            self.v0 = self.v[:, 0]

    def run_policy_and_value(self, sess, s_t):
        pi_out, v_out, logits = sess.run( [self.pi, self.v0, self.logits], feed_dict = {self.s : [s_t]} )
        return (pi_out[0], v_out[0], logits[0])

    def run_policy(self, sess, s_t):
        pi_out = sess.run( self.pi, feed_dict = {self.s : [s_t]} )
        return pi_out[0]

    def run_value(self, sess, s_t):
        v_out = sess.run( self.v0, feed_dict = {self.s : [s_t]} )
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
        self.last_hidden_fc_output_size = 512

        # state (input)
        self.s = tf.placeholder(tf.float32, [None, 84, 84, 4], name="state")
        self.s_n = tf.div(self.s, 255.)

        # place holder for LSTM unrolling time step size.
        self.step_size = tf.placeholder(tf.float32, [1])

        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            if self.use_mnih_2015:
                self.W_conv1, self.b_conv1 = self.conv_variable([8, 8, 4, 32], layer_name='conv1', gain=np.sqrt(2))
                self.W_conv2, self.b_conv2 = self.conv_variable([4, 4, 32, 64], layer_name='conv2', gain=np.sqrt(2))
                self.W_conv3, self.b_conv3 = self.conv_variable([3, 3, 64, 64], layer_name='conv3', gain=np.sqrt(2))
                self.W_fc1, self.b_fc1 = self.fc_variable([3136, self.last_hidden_fc_output_size], layer_name='fc1', gain=np.sqrt(2))
            else:
                self.W_conv1, self.b_conv1 = self.conv_variable([8, 8, 4, 16], layer_name='conv1', gain=np.sqrt(2))
                self.W_conv2, self.b_conv2 = self.conv_variable([4, 4, 16, 32], layer_name='conv2', gain=np.sqrt(2))
                self.W_fc1, self.b_fc1 = self.fc_variable([2592, self.last_hidden_fc_output_size], layer_name='fc1', gain=np.sqrt(2))

            # lstm
            self.lstm = tf.nn.rnn_cell.LSTMCell(self.last_hidden_fc_output_size, name='basic_lstm_cell', initializer=tf.orthogonal_initializer(gain=np.sqrt(2)))

            # weight for policy output layer
            self.W_fc2, self.b_fc2 = self.fc_variable([self.last_hidden_fc_output_size, action_size], layer_name='fc2')

            # weight for value output layer
            self.W_fc3, self.b_fc3 = self.fc_variable([self.last_hidden_fc_output_size, 1], layer_name='fc3')

            if self.use_mnih_2015:
                self.h_conv1 = tf.nn.relu(self.conv2d(self.s_n,  self.W_conv1, 4) + self.b_conv1)
                self.h_conv2 = tf.nn.relu(self.conv2d(self.h_conv1, self.W_conv2, 2) + self.b_conv2)
                self.h_conv3 = tf.nn.relu(self.conv2d(self.h_conv2, self.W_conv3, 1) + self.b_conv3)

                h_conv3_flat = tf.reshape(self.h_conv3, [-1, 3136])
                self.h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.W_fc1) + self.b_fc1)
            else:
                self.h_conv1 = tf.nn.relu(self.conv2d(self.s_n, self.W_conv1, 4) + self.b_conv1) # stride=4
                self.h_conv2 = tf.nn.relu(self.conv2d(self.h_conv1, self.W_conv2, 2) + self.b_conv2) # stride=2

                h_conv2_flat = tf.reshape(self.h_conv2, [-1, 2592])
                self.h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

            h_fc1_reshaped = tf.reshape(self.h_fc1, [1, -1, self.last_hidden_fc_output_size])

            self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, self.last_hidden_fc_output_size])
            self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, self.last_hidden_fc_output_size])
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

            # lstm_outputs: (1,5,fc_output_size) for back prop, (1,1,fc_output_size) for forward prop.
            lstm_outputs = tf.reshape(lstm_outputs, [-1, self.last_hidden_fc_output_size])

            # policy (output)
            # tf.shape(logits) [1, 6]
            self.logits = tf.matmul(lstm_outputs, self.W_fc2) + self.b_fc2
            self.pi = tf.nn.softmax(self.logits)

            # value (output)
            self.v = tf.matmul(lstm_outputs, self.W_fc3) + self.b_fc3
            self.v0 = self.v[:, 0]

            scope.reuse_variables()
            self.W_lstm = tf.get_variable("basic_lstm_cell/kernel")
            self.b_lstm = tf.get_variable("basic_lstm_cell/bias")

            self.reset_state()

    def reset_state(self):
        self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(
            np.zeros([1, self.last_hidden_fc_output_size]),
            np.zeros([1, self.last_hidden_fc_output_size]))

    def run_policy_and_value(self, sess, s_t):
        # This run_policy_and_value() is used when forward propagating.
        # so the step size is 1.
        pi_out, v_out, self.lstm_state_out, logits = sess.run(
            [self.pi, self.v0, self.lstm_state, self.logits],
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
        # When next sequence starts, V will be calculated again with the same state using updated network weights,
        # so we don't update LSTM state here.
        prev_lstm_state_out = self.lstm_state_out
        v_out, _ = sess.run(
            [self.v0, self.lstm_state],
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
