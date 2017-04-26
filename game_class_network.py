# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from termcolor import colored

# Base Class
class GameClassNetwork(object):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0"):
    self._action_size = action_size
    self._thread_index = thread_index
    self._device = device

  def prepare_loss(self):
    raise NotImplementedError()

  def prepare_evaluate(self):
    with tf.device(self._device):
      correct_prediction = tf.equal(tf.argmax(self._pi, 1), tf.argmax(self.a, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


  def run_policy_and_value(self, sess, s_t):
    raise NotImplementedError()

  def run_policy(self, sess, s_t):
    raise NotImplementedError()

  def run_value(self, sess, s_t):
    raise NotImplementedError()

  def get_vars(self):
    raise NotImplementedError()

  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "GameACNetwork", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  # weight initialization based on muupan's code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
  def _fc_variable(self, weight_shape, layer_name=''):
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d), name=layer_name + '_weights')
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d), name=layer_name + '_biases')
    return weight, bias

  def _conv_variable(self, weight_shape, layer_name=''):
    w = weight_shape[0]
    h = weight_shape[1]
    input_channels  = weight_shape[2]
    output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d), name=layer_name + '_weights')
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d), name=layer_name + '_biases')
    return weight, bias

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

# Actor-Critic FF Network
class GameACFFNetwork(GameClassNetwork):
  use_mnih_2015 = False
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0"):
    GameClassNetwork.__init__(self, action_size, thread_index, device)
    print (colored("use_mnih_2015: {}".format(self.use_mnih_2015), "green" if self.use_mnih_2015 else "red"))
    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      if self.use_mnih_2015:
        self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 32], layer_name='conv1')
        self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 32, 64], layer_name='conv2')
        self.W_conv3, self.b_conv3 = self._conv_variable([3, 3, 64, 64], layer_name='conv3')
        self.W_fc1, self.b_fc1 = self._fc_variable([3136, 256], layer_name='fc1')
        tf.add_to_collection('transfer_params', self.W_conv1)
        tf.add_to_collection('transfer_params', self.b_conv1)
        tf.add_to_collection('transfer_params', self.W_conv2)
        tf.add_to_collection('transfer_params', self.b_conv2)
        tf.add_to_collection('transfer_params', self.W_conv3)
        tf.add_to_collection('transfer_params', self.b_conv3)
        tf.add_to_collection('transfer_params', self.W_fc1)
        tf.add_to_collection('transfer_params', self.b_fc1)
      else:
        self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16], layer_name='conv1')  # stride=4
        self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32], layer_name='conv2') # stride=2
        self.W_fc1, self.b_fc1 = self._fc_variable([2592, 256], layer_name='fc1')
        tf.add_to_collection('transfer_params', self.W_conv1)
        tf.add_to_collection('transfer_params', self.b_conv1)
        tf.add_to_collection('transfer_params', self.W_conv2)
        tf.add_to_collection('transfer_params', self.b_conv2)
        tf.add_to_collection('transfer_params', self.W_fc1)
        tf.add_to_collection('transfer_params', self.b_fc1)

      # weight for policy output layer
      self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size], layer_name='fc2')
      tf.add_to_collection('transfer_params', self.W_fc2)
      tf.add_to_collection('transfer_params', self.b_fc2)

      # state (input)
      self.s = tf.placeholder("float", [None, 84, 84, 4])

      if self.use_mnih_2015:
        h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 4) + self.b_conv1)
        h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)
        h_conv3 = tf.nn.relu(self._conv2d(h_conv2, self.W_conv3, 1) + self.b_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 3136])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.W_fc1) + self.b_fc1)
      else:
        h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 4) + self.b_conv1)
        h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

        h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

      # policy (output)
      self._pi = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2
      self.pi = tf.nn.softmax(self._pi)

      self.max_value = tf.reduce_max(self._pi, axis=None)

  def prepare_loss(self):
    with tf.device(self._device):
      # taken action (input for policy)
      self.a = tf.placeholder(tf.float32, shape=[None, self._action_size])
      cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
          _sentinel=None,
          labels=self.a,
          logits=self._pi))
      self.total_loss = cross_entropy

  def run_policy_and_value(self, sess, s_t):
    raise NotImplementedError()

  def run_policy(self, sess, s_t):
    pi_out = sess.run( self.pi, feed_dict = {self.s : [s_t]} )
    return pi_out[0]

  def run_value(self, sess, s_t):
    raise NotImplementedError()

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

# Actor-Critic LSTM Network
class GameACLSTMNetwork(GameClassNetwork):
  use_mnih_2015 = False

  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0" ):
    GameClassNetwork.__init__(self, action_size, thread_index, device)
    print (colored("use_mnih_2015: {}".format(self.use_mnih_2015), "green" if self.use_mnih_2015 else "red"))
    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      if self.use_mnih_2015:
        self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 32], layer_name='conv1')
        self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 32, 64], layer_name='conv2')
        self.W_conv3, self.b_conv3 = self._conv_variable([3, 3, 64, 64], layer_name='conv3')
        self.W_fc1, self.b_fc1 = self._fc_variable([3136, 256], layer_name='fc1')
        tf.add_to_collection('transfer_params', self.W_conv1)
        tf.add_to_collection('transfer_params', self.b_conv1)
        tf.add_to_collection('transfer_params', self.W_conv2)
        tf.add_to_collection('transfer_params', self.b_conv2)
        tf.add_to_collection('transfer_params', self.W_conv3)
        tf.add_to_collection('transfer_params', self.b_conv3)
        tf.add_to_collection('transfer_params', self.W_fc1)
        tf.add_to_collection('transfer_params', self.b_fc1)
      else:
        self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16], layer_name='conv1')  # stride=4
        self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32], layer_name='conv2') # stride=2
        self.W_fc1, self.b_fc1 = self._fc_variable([2592, 256], layer_name='fc1')
        tf.add_to_collection('transfer_params', self.W_conv1)
        tf.add_to_collection('transfer_params', self.b_conv1)
        tf.add_to_collection('transfer_params', self.W_conv2)
        tf.add_to_collection('transfer_params', self.b_conv2)
        tf.add_to_collection('transfer_params', self.W_fc1)
        tf.add_to_collection('transfer_params', self.b_fc1)

      # lstm
      self.lstm = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

      # weight for policy output layer
      self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size], layer_name='fc2')
      tf.add_to_collection('transfer_params', self.W_fc2)
      tf.add_to_collection('transfer_params', self.b_fc2)

      # state (input)
      self.s = tf.placeholder("float", [None, 84, 84, 4])

      if self.use_mnih_2015:
        h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 4) + self.b_conv1)
        h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)
        h_conv3 = tf.nn.relu(self._conv2d(h_conv2, self.W_conv3, 1) + self.b_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 3136])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.W_fc1) + self.b_fc1)
      else:
        h_conv1 = tf.nn.relu(self._conv2d(self.s, self.W_conv1, 4) + self.b_conv1)
        h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

        h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

      h_fc1_reshaped = tf.reshape(h_fc1, [1,-1,256])

      # place holder for LSTM unrolling time step size.
      self.step_size = tf.placeholder(tf.float32, [1])

      self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, 256])
      self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, 256])
      self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0,
                                                              self.initial_lstm_state1)

      # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
      # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
      # Unrolling step size is applied via self.step_size placeholder.
      # When forward propagating, step_size is 1.
      # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
      lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                        h_fc1_reshaped,
                                                        initial_state = self.initial_lstm_state,
                                                        sequence_length = self.step_size,
                                                        time_major = False,
                                                        scope = scope)

      # lstm_outputs: (1,5,256) for back prop, (1,1,256) for forward prop.

      lstm_outputs = tf.reshape(lstm_outputs, [-1,256])

      # policy (output)
      self._pi = tf.matmul(lstm_outputs, self.W_fc2) + self.b_fc2
      self.pi = tf.nn.softmax(self._pi)

      self.max_value = tf.reduce_max(self._pi, axis=None)

      scope.reuse_variables()
      self.W_lstm = tf.get_variable("basic_lstm_cell/weights")
      self.b_lstm = tf.get_variable("basic_lstm_cell/biases")
      tf.add_to_collection('transfer_params', self.W_lstm)
      tf.add_to_collection('transfer_params', self.b_lstm)

      self.reset_state()

  def reset_state(self):
    self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 256]),
                                                        np.zeros([1, 256]))

  def prepare_loss(self):
    with tf.device(self._device):
      # taken action (input for policy)
      self.a = tf.placeholder(tf.float32, shape=[None, self._action_size])
      #   cross_entropy = tf.reduce_mean(
      #     tf.nn.softmax_cross_entropy_with_logits(
      #       labels=self.a,
      #       logits=self._pi))
      cross_entropy = tf.reduce_sum( tf.multiply( tf.log(self.pi), self.a ), reduction_indices=1 )
      self.total_loss = -tf.reduce_sum(cross_entropy)

  def run_policy_and_value(self, sess, s_t):
    raise NotImplementedError()

  def run_policy(self, sess, s_t):
    # This run_policy() is used for displaying the result with display tool.
    pi_out, self.lstm_state_out = sess.run( [self.pi, self.lstm_state],
                                            feed_dict = {self.s : [s_t],
                                                         self.initial_lstm_state0 : self.lstm_state_out[0],
                                                         self.initial_lstm_state1 : self.lstm_state_out[1],
                                                         self.step_size : [1]} )

    return pi_out[0]

  def run_value(self, sess, s_t):
    raise NotImplementedError()

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
