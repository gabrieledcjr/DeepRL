# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from abc import ABC, abstractmethod
from time import sleep
from termcolor import colored

# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(ABC):
  use_mnih_2015 = False
  l1_beta = 0.
  l2_beta = 0.
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
      self.a = tf.placeholder("float", [None, self._action_size])

      # temporary difference (R-V) (input for policy)
      self.td = tf.placeholder("float", [None])

      self.clip_min = tf.placeholder(tf.float32, shape=(), name="clip_minimum")
      # avoid NaN with clipping when value in pi becomes zero
      #log_pi = tf.log(tf.clip_by_value(self.pi, self.clip_min, 1.0))
      log_pi = tf.nn.log_softmax(tf.clip_by_value(self.logits, -10.0, 10.0))

      # policy entropy
      entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)

      # net_vars = self.get_vars()
      # l2_losses = []
      # l1_losses = []
      # for i in range(len(net_vars)):
      # if i%2 == 0:
      #   l1_losses.append(self.l1_beta * tf.reduce_sum(tf.abs(net_vars[i])))
      #   l2_losses.append(self.l2_beta * tf.nn.l2_loss(net_vars[i]))
      # l1_loss = sum(l1_losses)
      # l2_loss = sum(l2_losses)
      # l_losses = l1_loss + l2_loss

      self.policy_lr = tf.placeholder(tf.float32, shape=(), name="policy_lr")
      self.critic_lr = tf.placeholder(tf.float32, shape=(), name="critic_lr")
      self.entropy_beta = tf.placeholder(tf.float32, shape=(), name="entropy_beta")

      # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
      policy_loss = self.policy_lr * - tf.reduce_sum( tf.reduce_sum( tf.multiply(log_pi, self.a), axis=1 ) * self.td + entropy * self.entropy_beta)
      # policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.multiply( log_pi, self.a ) + l_losses, axis=1 ) * self.td + entropy * entropy_beta)

      # R (input for value)
      self.r = tf.placeholder("float", [None])

      # value loss (output)
      # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
      value_loss = self.critic_lr * tf.nn.l2_loss(self.r - self.v)

      # gradient of policy and value are summed up
      self.total_loss = policy_loss + value_loss

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

  def sync_from(self, src_network, name=None):
    src_vars = src_network.get_vars()
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

  def load_transfer_model(
    self, sess, folder='',
    not_transfer_fc2=False, not_transfer_fc1=False,
    not_transfer_conv3=False, not_transfer_conv2=False,
    var_list=None):
    assert folder != ''
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
      print (colored("Successfully loaded: {}".format(checkpoint_transfer_from.model_checkpoint_path), "green"))

      global_vars = tf.global_variables()
      is_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
      initialized_vars = [v for (v, f) in zip(global_vars, is_initialized) if f]
      for var in initialized_vars:
        print(colored("\t{} loaded".format(var.op.name), "green"))
        sleep(.5)

      if transfer_all:
        # scale down last layer if it's transferred
        print (colored("Normalizing output layer with max value {}...".format(transfer_max_output_val), "yellow"))
        W_fc2_norm = tf.div(self.W_fc2, transfer_max_output_val)
        b_fc2_norm = tf.div(self.b_fc2, transfer_max_output_val)
        print (colored("Output layer normalized", "green"))
        sess.run([
          self.W_fc2.assign(W_fc2_norm), self.b_fc2.assign(b_fc2_norm)
        ])

      sleep(2)

# Actor-Critic FF Network
class GameACFFNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0"):
    GameACNetwork.__init__(self, action_size, thread_index, device)
    print (colored("use_mnih_2015: {}".format(self.use_mnih_2015), "green" if self.use_mnih_2015 else "red"))
    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      if self.use_mnih_2015:
        self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 32], layer_name='conv1')
        self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 32, 64], layer_name='conv2')
        self.W_conv3, self.b_conv3 = self._conv_variable([3, 3, 64, 64], layer_name='conv3')
        self.W_fc1, self.b_fc1 = self._fc_variable([3136, 256], layer_name='fc1')
      else:
        self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16], layer_name='conv1')  # stride=4
        self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32], layer_name='conv2') # stride=2
        self.W_fc1, self.b_fc1 = self._fc_variable([2592, 256], layer_name='fc1')

      # weight for policy output layer
      self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size], layer_name='fc2')

      # weight for value output layer
      self.W_fc3, self.b_fc3 = self._fc_variable([256, 1], layer_name='fc3')

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

  def get_vars_pi(self):
    if self.use_mnih_2015:
      return [self.W_conv1, self.b_conv1,
              self.W_conv2, self.b_conv2,
              self.W_conv3, self.b_conv3,
              self.W_fc1, self.b_fc1,
              self.W_fc2, self.b_fc2]
    else:
      return [self.W_conv1, self.b_conv1,
              self.W_conv2, self.b_conv2,
              self.W_fc1, self.b_fc1,
              self.W_fc2, self.b_fc2]

  def get_vars_v(self):
    if self.use_mnih_2015:
      return [self.W_conv1, self.b_conv1,
              self.W_conv2, self.b_conv2,
              self.W_conv3, self.b_conv3,
              self.W_fc1, self.b_fc1,
              self.W_fc3, self.b_fc3]
    else:
      return [self.W_conv1, self.b_conv1,
              self.W_conv2, self.b_conv2,
              self.W_fc1, self.b_fc1,
              self.W_fc3, self.b_fc3]

# Actor-Critic LSTM Network
class GameACLSTMNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0"):
    GameACNetwork.__init__(self, action_size, thread_index, device)
    print (colored("use_mnih_2015: {}".format(self.use_mnih_2015), "green" if self.use_mnih_2015 else "red"))
    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      if self.use_mnih_2015:
        self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 32], layer_name='conv1')
        self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 32, 64], layer_name='conv2')
        self.W_conv3, self.b_conv3 = self._conv_variable([3, 3, 64, 64], layer_name='conv3')
        self.W_fc1, self.b_fc1 = self._fc_variable([3136, 256], layer_name='fc1')
      else:
        self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16], layer_name='conv1')  # stride=4
        self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32], layer_name='conv2') # stride=2
        self.W_fc1, self.b_fc1 = self._fc_variable([2592, 256], layer_name='fc1')

      # lstm
      self.lstm = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

      # weight for policy output layer
      self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size], layer_name='fc2')

      # weight for value output layer
      self.W_fc3, self.b_fc3 = self._fc_variable([256, 1], layer_name='fc3')

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
      self.logits = tf.matmul(lstm_outputs, self.W_fc2) + self.b_fc2
      self.pi = tf.nn.softmax(self.logits)

      # value (output)
      v_ = tf.matmul(lstm_outputs, self.W_fc3) + self.b_fc3
      self.v = tf.reshape( v_, [-1] )

      scope.reuse_variables()
      self.W_lstm = tf.get_variable("basic_lstm_cell/weights")
      self.b_lstm = tf.get_variable("basic_lstm_cell/biases")

      self.reset_state()

  def reset_state(self):
    self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 256]),
                                                        np.zeros([1, 256]))

  def run_policy_and_value(self, sess, s_t):
    # This run_policy_and_value() is used when forward propagating.
    # so the step size is 1.
    pi_out, v_out, self.lstm_state_out, logits = sess.run( [self.pi, self.v, self.lstm_state, self.logits],
                                                               feed_dict = {self.s : [s_t],
                                                                            self.initial_lstm_state0 : self.lstm_state_out[0],
                                                                            self.initial_lstm_state1 : self.lstm_state_out[1],
                                                                            self.step_size : [1]} )
    # pi_out: (1,3), v_out: (1)
    return (pi_out[0], v_out[0], logits[0])

  def run_policy(self, sess, s_t):
    # This run_policy() is used for displaying the result with display tool.
    pi_out, self.lstm_state_out = sess.run( [self.pi, self.lstm_state],
                                            feed_dict = {self.s : [s_t],
                                                         self.initial_lstm_state0 : self.lstm_state_out[0],
                                                         self.initial_lstm_state1 : self.lstm_state_out[1],
                                                         self.step_size : [1]} )

    return pi_out[0]

  def run_value(self, sess, s_t):
    # This run_value() is used for calculating V for bootstrapping at the
    # end of LOCAL_T_MAX time step sequence.
    # When next sequcen starts, V will be calculated again with the same state using updated network weights,
    # so we don't update LSTM state here.
    prev_lstm_state_out = self.lstm_state_out
    v_out, _ = sess.run( [self.v, self.lstm_state],
                         feed_dict = {self.s : [s_t],
                                      self.initial_lstm_state0 : self.lstm_state_out[0],
                                      self.initial_lstm_state1 : self.lstm_state_out[1],
                                      self.step_size : [1]} )

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

  def get_vars_pi(self):
    if self.use_mnih_2015:
      return [self.W_conv1, self.b_conv1,
              self.W_conv2, self.b_conv2,
              self.W_conv3, self.b_conv3,
              self.W_fc1, self.b_fc1,
              self.W_lstm, self.b_lstm,
              self.W_fc2, self.b_fc2]
    else:
      return [self.W_conv1, self.b_conv1,
              self.W_conv2, self.b_conv2,
              self.W_fc1, self.b_fc1,
              self.W_lstm, self.b_lstm,
              self.W_fc2, self.b_fc2]

  def get_vars_v(self):
    if self.use_mnih_2015:
      return [self.W_conv1, self.b_conv1,
              self.W_conv2, self.b_conv2,
              self.W_conv3, self.b_conv3,
              self.W_fc1, self.b_fc1,
              self.W_lstm, self.b_lstm,
              self.W_fc3, self.b_fc3]
    else:
      return [self.W_conv1, self.b_conv1,
              self.W_conv2, self.b_conv2,
              self.W_fc1, self.b_fc1,
              self.W_lstm, self.b_lstm,
              self.W_fc3, self.b_fc3]
