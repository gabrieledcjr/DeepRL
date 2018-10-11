#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import time
import logging

from termcolor import colored
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from common.game_state import GameState, get_wrapper_by_name
from common.util import get_action_index

logger = logging.getLogger("a3c_training_thread")

class A3CTrainingThread(object):
    log_interval = 100
    performance_log_interval = 1000
    local_t_max = 20
    demo_t_max = 20
    use_lstm = False
    action_size = -1
    entropy_beta = 0.01
    demo_entropy_beta = 0.01
    gamma = 0.99
    use_mnih_2015 = False
    env_id = None
    log_scale_reward = False
    egreedy_testing = False
    finetune_upper_layers_oinly = False
    shaping_reward = 0.001
    shaping_factor = 1.
    shaping_gamma = 0.85
    advice_confidence = 0.8
    shaping_actions = -1 # -1 all actions, 0 exclude noop

    def __init__(self,
                 thread_index,
                 global_network,
                 initial_learning_rate,
                 learning_rate_input,
                 grad_applier,
                 max_global_time_step,
                 device=None,
                 pretrained_model=None,
                 pretrained_model_sess=None,
                 advice=False,
                 reward_shaping=False):
        assert self.action_size != -1

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step
        self.use_pretrained_model_as_advice = advice
        self.use_pretrained_model_as_reward_shaping = reward_shaping

        logger.info("thread_index: {}".format(self.thread_index))
        logger.info("local_t_max: {}".format(self.local_t_max))
        logger.info("use_lstm: {}".format(colored(self.use_lstm, "green" if self.use_lstm else "red")))
        logger.info("action_size: {}".format(self.action_size))
        logger.info("entropy_beta: {}".format(self.entropy_beta))
        logger.info("gamma: {}".format(self.gamma))
        logger.info("log_scale_reward: {}".format(colored(self.log_scale_reward, "green" if self.log_scale_reward else "red")))
        logger.info("egreedy_testing: {}".format(colored(self.egreedy_testing, "green" if self.egreedy_testing else "red")))
        logger.info("finetune_upper_layers_only: {}".format(colored(self.finetune_upper_layers_only, "green" if self.finetune_upper_layers_only else "red")))
        logger.info("use_pretrained_model_as_advice: {}".format(colored(self.use_pretrained_model_as_advice, "green" if self.use_pretrained_model_as_advice else "red")))
        logger.info("use_pretrained_model_as_reward_shaping: {}".format(colored(self.use_pretrained_model_as_reward_shaping, "green" if self.use_pretrained_model_as_reward_shaping else "red")))

        if self.use_lstm:
            GameACLSTMNetwork.use_mnih_2015 = self.use_mnih_2015
            self.local_network = GameACLSTMNetwork(self.action_size, thread_index, device)
        else:
            GameACFFNetwork.use_mnih_2015 = self.use_mnih_2015
            self.local_network = GameACFFNetwork(self.action_size, thread_index, device)

        self.local_network.prepare_loss()

        with tf.device(device):
            local_vars = self.local_network.get_vars
            if self.finetune_upper_layers_only:
                local_vars = self.local_network.get_vars_upper
            var_refs = [v._ref() for v in local_vars()]

            self.gradients = tf.gradients(
                self.local_network.total_loss, var_refs,
                gate_gradients=False,
                aggregation_method=None,
                colocate_gradients_with_ops=False)

        global_vars = global_network.get_vars
        if self.finetune_upper_layers_only:
            global_vars = global_network.get_vars_upper

        self.apply_gradients = grad_applier.apply_gradients(
            global_vars(),
            self.gradients)

        self.sync = self.local_network.sync_from(
            global_network, upper_layers_only=self.finetune_upper_layers_only)

        self.game_state = GameState(env_id=self.env_id, display=False, no_op_max=30, human_demo=False, episode_life=True)

        self.local_t = 0

        self.initial_learning_rate = initial_learning_rate

        self.episode_reward = 0
        self.episode_steps = 0

        # variable controling log output
        self.prev_local_t = 0

        self.is_demo_thread = False
        self.is_egreedy = False

        self.pretrained_model = pretrained_model
        self.pretrained_model_sess = pretrained_model_sess
        self.psi = 0.9 if self.use_pretrained_model_as_advice else 0.0
        self.advice_ctr = 0
        self.shaping_ctr = 0
        self.last_rho = 0.

        if self.use_pretrained_model_as_advice or self.use_pretrained_model_as_reward_shaping:
            assert self.pretrained_model is not None

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, pi_values):
        return np.random.choice(range(self.action_size), p=pi_values)

    def choose_action_egreedy(self, pi_values, global_time_step):
        if global_time_step < 1000000:
            epsilon = 0.2 * (global_time_step//200000)
            epsilon = 0.1 if epsilon == 0.0 else epsilon+0.1
            if np.random.random() > epsilon:
                return get_action_index(pi_values, is_random=False, n_actions=self.action_size)
        return self.choose_action(pi_values)

    def choose_action_with_high_confidence(self, pi_values, exclude_noop=True):
        actions_confidence = []
        # exclude NOOP action
        for action in range(1 if exclude_noop else 0, self.action_size):
            actions_confidence.append(pi_values[action][0][0])
        max_confidence_action = np.argmax(actions_confidence)
        confidence = actions_confidence[max_confidence_action]
        return (max_confidence_action+(1 if exclude_noop else 0)), confidence

    def _record_summary(self, sess, summary_writer, summary_op, score_input, steps_input, score, steps, global_t):
        summary_str = sess.run(summary_op, feed_dict={
            score_input: float(score),
            steps_input: steps
        })
        summary_writer.add_summary(summary_str, global_t)
        summary_writer.flush()

    def set_start_time(self, start_time):
        self.start_time = start_time

    def testing(self, sess, max_steps, global_t, summary_writer):
        logger.info("Evaluate policy at global_t={}...".format(global_t))
        # copy weights from shared to local
        sess.run( self.sync )

        total_ep_rewards = 0
        total_ep_steps = 0
        total_steps = 0
        episode_count = 0
        while True:
            self.game_state.reset(hard_reset=True)
            if self.use_lstm:
                self.local_network.reset_state()

            episode_reward = 0
            episode_steps = 0
            while total_steps < max_steps:
                pi_ = self.local_network.run_policy(sess, self.game_state.s_t)
                if self.egreedy_testing:
                    action = get_action_index(pi_, is_random=(np.random.random() <= 0.1), n_actions=self.action_size)
                else:
                    action = self.choose_action(pi_)

                if self.use_pretrained_model_as_advice:
                    psi = self.psi if self.psi > 0.001 else 0.0
                    if psi > np.random.rand():
                        model_pi = self.pretrained_model.run_policy(self.pretrained_model_sess, self.game_state.s_t)
                        model_action, confidence = self.choose_action_with_high_confidence(model_pi, exclude_noop=False)
                        if (model_action > self.shaping_actions and confidence >= self.advice_confidence):
                            action = model_action

                # process game
                self.game_state.step(action)

                # receive game result
                reward = self.game_state.reward
                terminal = self.game_state.terminal
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                # s_t1 -> s_t
                self.game_state.update()

                if terminal or (episode_count == 0 and total_steps == max_steps):
                    if get_wrapper_by_name(self.game_state.env, 'EpisodicLifeEnv').was_real_done or \
                       (episode_count == 0 and total_steps == max_steps):
                        total_ep_rewards += episode_reward
                        total_ep_steps += episode_steps
                        episode_count += 1
                        score_str = colored("score={}".format(episode_reward), "magenta")
                        steps_str = colored("steps={}".format(episode_steps), "blue")
                        logger.debug("test: global_t={} t_idx={} total_steps={} {} {}".format(global_t, self.thread_index, total_steps, score_str, steps_str))
                        break

                    self.game_state.reset(hard_reset=False)
                    if self.use_lstm:
                        self.local_network.reset_state()

            if total_steps >= max_steps:
                logger.debug("test: global_t={} t_idx={} total_steps={} score={} steps={} max steps reached".format(global_t, self.thread_index, total_steps, episode_reward, episode_steps))
                break

        testing_reward = total_ep_rewards / episode_count
        testing_steps = total_ep_steps // episode_count
        logger.info("Test Evaluation: global_t={} t_idx={} final score={} final steps={}".format(global_t, self.thread_index, testing_reward, testing_steps))

        summary = tf.Summary()
        summary.value.add(tag='Testing/score', simple_value=float(testing_reward))
        summary.value.add(tag='Testing/steps', simple_value=testing_steps)
        summary_writer.add_summary(summary, global_t)
        summary_writer.flush()

        self.episode_reward = 0
        self.episode_steps = 0
        self.game_state.reset(hard_reset=True)
        self.last_rho = 0.
        if self.is_demo_thread:
            self.replay_mem_reset()

        if self.use_lstm:
            self.local_network.reset_state()
        return testing_reward, testing_steps

    def pretrain_init(self, demo_memory):
        self.demo_memory_size = len(demo_memory)
        self.demo_memory = demo_memory
        self.replay_mem_reset()

    def replay_mem_reset(self, demo_memory_idx=None):
        if demo_memory_idx is not None:
            self.demo_memory_idx = demo_memory_idx
        else:
            # new random episode
            self.demo_memory_idx = np.random.randint(0, self.demo_memory_size)
        self.demo_memory_count = np.random.randint(0, len(self.demo_memory[self.demo_memory_idx])-self.local_t_max)
        # if self.demo_memory_count+self.local_t_max < len(self.demo_memory[self.demo_memory_idx]):
        #           self.demo_memory_max_count = np.random.randint(self.demo_memory_count+self.local_t_max, len(self.demo_memory[self.demo_memory_idx]))
        # else:
        #           self.demo_memory_max_count = len(self.demo_memory[self.demo_memory_idx])
        logger.debug("t_idx={} mem_reset demo_memory_idx={} demo_memory_start={}".format(self.thread_index, self.demo_memory_idx, self.demo_memory_count))
        s_t, action, reward, terminal = self.demo_memory[self.demo_memory_idx][self.demo_memory_count]
        self.demo_memory_action = action
        self.demo_memory_reward = reward
        self.demo_memory_terminal = terminal
        if not self.demo_memory[self.demo_memory_idx].imgs_normalized:
            self.demo_memory_s_t = s_t * (1.0/255.0)
        else:
            self.demo_memory_s_t = s_t

    def replay_mem_process(self):
        self.demo_memory_count += 1
        s_t, action, reward, terminal = self.demo_memory[self.demo_memory_idx][self.demo_memory_count]
        self.demo_memory_next_action = action
        self.demo_memory_reward = reward
        self.demo_memory_terminal = terminal
        if not self.demo_memory[self.demo_memory_idx].imgs_normalized:
            self.demo_memory_s_t1 = s_t * (1.0/255.0)
        else:
            self.demo_memory_s_t1 = s_t

    def replay_mem_update(self):
        self.demo_memory_action = self.demo_memory_next_action
        self.demo_memory_s_t = self.demo_memory_s_t1

    def demo_process(self, sess, global_t, demo_memory_idx=None):
        states = []
        actions = []
        rewards = []
        values = []

        demo_ended = False
        terminal_end = False

        # copy weights from shared to local
        sess.run( self.sync )

        start_local_t = self.local_t

        if self.use_lstm:
            reset_lstm_state = False
            start_lstm_state = self.local_network.lstm_state_out

        # t_max times loop
        for i in range(self.demo_t_max):
            pi_, value_, logits_ = self.local_network.run_policy_and_value(sess, self.demo_memory_s_t)
            action = self.demo_memory_action
            time.sleep(0.0025)

            states.append(self.demo_memory_s_t)
            actions.append(action)
            values.append(value_)

            if (self.thread_index == 0) and (self.local_t % self.log_interval == 0):
                logger.debug("lg={}".format(np.array_str(logits_, precision=4, suppress_small=True)))
                logger.debug("pi={}".format(np.array_str(pi_, precision=4, suppress_small=True)))
                logger.debug("V={}".format(value_))

            # process replay memory
            self.replay_mem_process()

            # receive replay memory result
            reward = self.demo_memory_reward
            terminal = self.demo_memory_terminal

            self.episode_reward += reward

            if self.log_scale_reward:
                reward = np.sign(reward) * np.log(1 + np.abs(reward))
            else:
                # clip reward
                reward = np.sign(reward)

            rewards.append(reward)

            self.local_t += 1
            self.episode_steps += 1

            # demo_memory_s_t1 -> demo_memory_s_t
            self.replay_mem_update()
            s_t = self.demo_memory_s_t

            if terminal or self.demo_memory_count == len(self.demo_memory[self.demo_memory_idx]):
                logger.debug("t_idx={} score={}".format(self.thread_index, self.episode_reward))
                demo_ended = True
                if terminal:
                    terminal_end = True
                    if self.use_lstm:
                        self.local_network.reset_state()

                else:
                    # some demo episodes doesn't reach terminal state
                    if self.use_lstm:
                        reset_lstm_state = True

                self.episode_reward = 0
                self.episode_steps = 0
                self.replay_mem_reset(demo_memory_idx=demo_memory_idx)
                break

        R = 0.0
        if not terminal_end:
            R = self.local_network.run_value(sess, s_t)

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

        cur_learning_rate = self._anneal_learning_rate(global_t) #* 0.005

        if self.use_lstm:
            batch_si.reverse()
            batch_a.reverse()
            batch_td.reverse()
            batch_R.reverse()

            sess.run(self.apply_gradients,
                     feed_dict = {
                         self.local_network.s: batch_si,
                         self.local_network.a: batch_a,
                         self.local_network.td: batch_td,
                         self.local_network.r: batch_R,
                         self.local_network.policy_lr: 1.0,
                         self.local_network.critic_lr: 0.5,
                         self.local_network.entropy_beta: self.demo_entropy_beta,
                         self.local_network.initial_lstm_state: start_lstm_state,
                         self.local_network.step_size : [len(batch_a)],
                         self.learning_rate_input: cur_learning_rate} )

            # some demo episodes doesn't reach terminal state
            if reset_lstm_state:
                self.local_network.reset_state()
                reset_lstm_state = False
        else:
            sess.run(self.apply_gradients,
                     feed_dict = {
                         self.local_network.s: batch_si,
                         self.local_network.a: batch_a,
                         self.local_network.td: batch_td,
                         self.local_network.r: batch_R,
                         self.local_network.policy_lr: 1.0,
                         self.local_network.critic_lr: 0.5,
                         self.local_network.entropy_beta: self.demo_entropy_beta,
                         self.learning_rate_input: cur_learning_rate} )

        if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= self.performance_log_interval):
            self.prev_local_t += self.performance_log_interval

        # return advancd local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t, demo_ended

    def process(self, sess, global_t, summary_writer, summary_op, score_input, steps_input, train_rewards):
        states = []
        actions = []
        rewards = []
        values = []
        rho = []

        terminal_end = False

        # copy weights from shared to local
        sess.run( self.sync )

        start_local_t = self.local_t

        if self.use_lstm:
            start_lstm_state = self.local_network.lstm_state_out

        # t_max times loop
        for i in range(self.local_t_max):
            pi_, value_, logits_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
            if self.is_egreedy:
                action = self.choose_action_egreedy(pi_, global_t)
            else:
                action = self.choose_action(pi_)

            model_pi = None
            confidence = 0.
            if self.use_pretrained_model_as_advice:
                self.psi = 0.9999 * (0.9999 ** global_t) if self.psi > 0.001 else 0.0 # 0.99995 works
                if self.psi > np.random.rand():
                    model_pi = self.pretrained_model.run_policy(self.pretrained_model_sess, self.game_state.s_t)
                    model_action, confidence = self.choose_action_with_high_confidence(model_pi, exclude_noop=False)
                    if (model_action > self.shaping_actions and confidence >= self.advice_confidence):
                        action = model_action
                        self.advice_ctr += 1
            if self.use_pretrained_model_as_reward_shaping:
                #if action > 0:
                if model_pi is None:
                    model_pi = self.pretrained_model.run_policy(self.pretrained_model_sess, self.game_state.s_t)
                    confidence = model_pi[action][0][0]
                if (action > self.shaping_actions and confidence >= self.advice_confidence):
                    #rho.append(round(confidence, 5))
                    rho.append(self.shaping_reward)
                    self.shaping_ctr += 1
                else:
                    rho.append(0.)
                #self.shaping_ctr += 1

            states.append(self.game_state.s_t)
            actions.append(action)
            values.append(value_)

            if self.thread_index == 0 and self.local_t % self.log_interval == 0:
                log_msg = "lg={} ".format(np.array_str(logits_, precision=4, suppress_small=True))
                log_msg += "pi={} ".format(np.array_str(pi_, precision=4, suppress_small=True))
                log_msg += "V={:.4f} ".format(value_)
                if self.use_pretrained_model_as_advice:
                    log_msg += "psi={:.4f}".format(self.psi)
                logger.debug(log_msg)

            # process game
            self.game_state.step(action)

            # receive game result
            reward = self.game_state.reward
            terminal = self.game_state.terminal
            if self.use_pretrained_model_as_reward_shaping:
                if reward < 0 and reward > 0:
                    rho[i] = 0.
                    j = i-1
                    while j > i-5:
                        if rewards[j] != 0:
                            break
                        rho[j] = 0.
                        j -= 1
            #     if self.game_state.loss_life:
            #     if self.game_state.gain_life or reward > 0:
            #         rho[i] = 0.
            #         j = i-1
            #         k = 1
            #         while j >= 0:
            #             if rewards[j] != 0:
            #                 rho[j] = self.shaping_reward * (self.gamma ** -1)
            #                 break
            #             rho[j] = self.shaping_reward / k
            #             j -= 1
            #             k += 1

            self.episode_reward += reward

            if self.log_scale_reward:
                reward = np.sign(reward) * np.log(1 + np.abs(reward))
            else:
                reward = np.sign(reward)  # clip reward

            rewards.append(reward)

            self.local_t += 1
            self.episode_steps += 1

            # s_t1 -> s_t
            self.game_state.update()

            if terminal:
                if get_wrapper_by_name(self.game_state.env, 'EpisodicLifeEnv').was_real_done:
                    log_msg = "train: worker={} global_t={}".format(self.thread_index, global_t)
                    if self.use_pretrained_model_as_advice:
                        log_msg += " advice_ctr={}".format(self.advice_ctr)
                    if self.use_pretrained_model_as_reward_shaping:
                        log_msg += " shaping_ctr={}".format(self.shaping_ctr)
                    score_str = colored("score={}".format(self.episode_reward), "magenta")
                    steps_str = colored("steps={}".format(self.episode_steps), "blue")
                    log_msg += " {} {}".format(score_str, steps_str)
                    logger.debug(log_msg)
                    train_rewards['train'][global_t] = (self.episode_reward, self.episode_steps)
                    self._record_summary(
                        sess, summary_writer, summary_op,
                        score_input, steps_input,
                        self.episode_reward, self.episode_steps, global_t)
                    self.episode_reward = 0
                    self.episode_steps = 0

                terminal_end = True
                self.last_rho = 0.
                if self.use_lstm:
                    self.local_network.reset_state()
                self.game_state.reset(hard_reset=False)
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

        if self.use_pretrained_model_as_reward_shaping:
            rho.reverse()
            rho.append(self.last_rho)
            self.last_rho = rho[0]
            i = 0
            # compute and accumulate gradients
            for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
                # Wiewiora et al.(2003) Principled Methods for Advising RL agents
                # Look-Back Advice
                #F = rho[i] - (self.shaping_gamma**-1) * rho[i+1]
                #F = rho[i] - self.shaping_gamma * rho[i+1]
                F = (self.shaping_gamma**-1) * rho[i] - rho[i+1]
                if (i == 0 and terminal_end) or (F != 0 and (ri > 0 or ri < 0)):
                    #logger.warn("averted additional F in absorbing state")
                    F = 0.
                # if (F < 0. and ri > 0) or (F > 0. and ri < 0):
                #     logger.warn("Negative reward shaping F={} ri={} rho[s]={} rhos[s-1]={}".format(F, ri, rho[i], rho[i+1]))
                #     F = 0.
                R = (ri + F*self.shaping_factor) + self.gamma * R
                td = R - Vi

                a = np.zeros([self.action_size])
                a[ai] = 1

                batch_si.append(si)
                batch_a.append(a)
                batch_td.append(td)
                batch_R.append(R)
                i += 1
        else:
            # compute and accumulate gradients
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

            sess.run(self.apply_gradients,
                feed_dict = {
                    self.local_network.s: batch_si,
                    self.local_network.a: batch_a,
                    self.local_network.td: batch_td,
                    self.local_network.r: batch_R,
                    self.local_network.policy_lr: 1.0,
                    self.local_network.critic_lr: 0.5,
                    self.local_network.entropy_beta: self.entropy_beta,
                    self.local_network.initial_lstm_state: start_lstm_state,
                    self.local_network.step_size : [len(batch_a)],
                    self.learning_rate_input: cur_learning_rate} )
        else:
            sess.run(self.apply_gradients,
                feed_dict = {
                    self.local_network.s: batch_si,
                    self.local_network.a: batch_a,
                    self.local_network.td: batch_td,
                    self.local_network.r: batch_R,
                    self.local_network.policy_lr: 1.0,
                    self.local_network.critic_lr: 0.5,
                    self.local_network.entropy_beta: self.entropy_beta,
                    self.learning_rate_input: cur_learning_rate} )

        if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= self.performance_log_interval):
            self.prev_local_t += self.performance_log_interval
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            logger.info("Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t, terminal_end
