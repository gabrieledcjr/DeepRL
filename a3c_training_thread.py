# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys

from termcolor import colored
from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from util import get_action_index


class A3CTrainingThread(object):
  log_interval = 100
  performance_log_interval = 1000
  local_t_max = 5
  use_lstm = False
  action_size = -1
  entropy_beta = 0.01
  gamma = 0.99
  use_mnih_2015 = False
  env_id = None

  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device=None):
    assert self.action_size != -1

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step

    print (colored("local_t_max: {}".format(self.local_t_max), "green"))
    print (colored("use_lstm: {}".format(self.use_lstm), "green" if self.use_lstm else "red"))
    print (colored("action_size: {}".format(self.action_size), "green"))
    print (colored("entropy_beta: {}".format(self.entropy_beta), "green"))
    print (colored("gamma: {}".format(self.gamma), "green"))

    if self.use_lstm:
      GameACLSTMNetwork.use_mnih_2015 = self.use_mnih_2015
      self.local_network = GameACLSTMNetwork(self.action_size, thread_index, device)
    else:
      GameACFFNetwork.use_mnih_2015 = self.use_mnih_2015
      self.local_network = GameACFFNetwork(self.action_size, thread_index, device)

    self.local_network.prepare_loss(self.entropy_beta)

    with tf.device(device):
      var_refs = [v._ref() for v in self.local_network.get_vars()]
      self.gradients = tf.gradients(
        self.local_network.total_loss, var_refs,
        gate_gradients=False,
        aggregation_method=None,
        colocate_gradients_with_ops=False)

    self.apply_gradients = grad_applier.apply_gradients(
      global_network.get_vars(),
      self.gradients )

    self.sync = self.local_network.sync_from(global_network)

    self.game_state = GameState(env_id=self.env_id)

    self.local_t = 0

    self.initial_learning_rate = initial_learning_rate

    self.episode_reward = 0

    # variable controling log output
    self.prev_local_t = 0

  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      score_input: float(score)
    })
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()

  def set_start_time(self, start_time):
    self.start_time = start_time

  def testing(self, sess, max_steps, global_t, summary_writer):
    # copy weights from shared to local
    sess.run( self.sync )

    self.game_state.reset()
    if self.use_lstm:
      self.local_network.reset_state()

    total_rewards = 0.
    total_steps = 0.
    episode_count = 0
    while True:
      if self.use_lstm:
        start_lstm_state = self.local_network.lstm_state_out

      episode_reward = 0
      while total_steps < max_steps:
        pi_ = self.local_network.run_policy(sess, self.game_state.s_t)
        #action = self.choose_action(pi_)
        action = get_action_index(pi_, is_random=(np.random.random() <= 0.05), n_actions=self.game_state.env.n_actions)

        # process game
        self.game_state.process(action)

        # receive game result
        reward = self.game_state.reward
        terminal = self.game_state.terminal

        episode_reward += reward
        total_steps += 1

        # s_t1 -> s_t
        self.game_state.update()

        if terminal:
          total_rewards += episode_reward
          episode_count += 1
          print(colored("test: global_t={} t_idx={} score={} total_steps={}".format(global_t, self.thread_index, episode_reward, total_steps), "yellow"))

          self.game_state.reset()
          if self.use_lstm:
            self.local_network.reset_state()
          break

      if total_steps >= max_steps:
          break

    testing_reward = total_rewards / episode_count
    print(colored("test: global_t={} t_idx={} final score={}".format(global_t, self.thread_index, testing_reward), "green"))

    summary = tf.Summary()
    summary.value.add(tag='Testing/score', simple_value=float(testing_reward))
    summary_writer.add_summary(summary, global_t)
    summary_writer.flush()

    self.episode_reward = 0
    self.game_state.reset()
    if self.use_lstm:
      self.local_network.reset_state()
    return testing_reward

  def process(self, sess, global_t, summary_writer, summary_op, score_input):
    states = []
    actions = []
    rewards = []
    values = []

    terminal_end = False

    # copy weights from shared to local
    sess.run( self.sync )

    start_local_t = self.local_t

    if self.use_lstm:
      start_lstm_state = self.local_network.lstm_state_out

    # t_max times loop
    for i in range(self.local_t_max):
      pi_, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
      action = self.choose_action(pi_)

      states.append(self.game_state.s_t)
      actions.append(action)
      values.append(value_)

      if (self.thread_index == 0) and (self.local_t % self.log_interval == 0):
        print("pi={}".format(pi_))
        print(" V={}".format(value_))

      # process game
      self.game_state.process(action)

      # receive game result
      reward = self.game_state.reward
      terminal = self.game_state.terminal

      self.episode_reward += reward

      # clip reward
      rewards.append( np.clip(reward, -1, 1) )

      self.local_t += 1

      # s_t1 -> s_t
      self.game_state.update()

      if terminal:
        terminal_end = True
        print("t_idx={} score={}".format(self.thread_index, self.episode_reward))

        self._record_score(sess, summary_writer, summary_op, score_input,
                           self.episode_reward, global_t)

        self.episode_reward = 0
        self.game_state.reset()
        if self.use_lstm:
          self.local_network.reset_state()
        break

    R = 0.0
    if not terminal_end:
      R = self.local_network.run_value(sess, self.game_state.s_t)

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si = []
    batch_a = []
    batch_td = []
    batch_R = []

    # compute and accmulate gradients
    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + self.gamma * R
      td = R - Vi
      a = np.zeros([self.action_size])
      a[ai] = 1

      batch_si.append(si)
      batch_a.append(a)
      batch_td.append(td)
      batch_R.append(R)

    cur_learning_rate = self._anneal_learning_rate(global_t)

    if self.use_lstm:
      batch_si.reverse()
      batch_a.reverse()
      batch_td.reverse()
      batch_R.reverse()

      sess.run( self.apply_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_a,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.local_network.initial_lstm_state: start_lstm_state,
                  self.local_network.step_size : [len(batch_a)],
                  self.learning_rate_input: cur_learning_rate } )
    else:
      sess.run( self.apply_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_a,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.learning_rate_input: cur_learning_rate} )

    if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= self.performance_log_interval):
      self.prev_local_t += self.performance_log_interval
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      print(colored("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
        global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.), "blue"))

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t
