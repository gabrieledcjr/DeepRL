#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import time
import logging

from termcolor import colored
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from common.game_state import GameState, get_wrapper_by_name
from common.util import make_movie, visualize_cam, generate_image_for_cam_video

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
    reward_type = 'CLIP'  # CLIP | LOG | RAW
    finetune_upper_layers_oinly = False
    shaping_reward = 0.001
    shaping_factor = 1.
    shaping_gamma = 0.85
    advice_confidence = 0.8
    shaping_actions = -1  # -1 all actions, 0 exclude noop
    transformed_bellman = False
    clip_norm = 0.5
    load_demo_cam = False

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
        logger.info("reward_type: {}".format(self.reward_type))
        logger.info("finetune_upper_layers_only: {}".format(
            colored(self.finetune_upper_layers_only, "green" if self.finetune_upper_layers_only else "red")))
        logger.info("use_pretrained_model_as_advice: {}".format(
            colored(self.use_pretrained_model_as_advice, "green" if self.use_pretrained_model_as_advice else "red")))
        logger.info("use_pretrained_model_as_reward_shaping: {}".format(
            colored(self.use_pretrained_model_as_reward_shaping,
                    "green" if self.use_pretrained_model_as_reward_shaping else "red")))
        logger.info("transformed_bellman: {}".format(
            colored(self.transformed_bellman, "green" if self.transformed_bellman else "red")))
        logger.info("clip_norm: {}".format(self.clip_norm))
        logger.info("load_demo_cam: {}".format(
            colored(self.load_demo_cam, "green" if self.load_demo_cam else "red")))

        if self.use_lstm:
            GameACLSTMNetwork.use_mnih_2015 = self.use_mnih_2015
            self.local_network = GameACLSTMNetwork(self.action_size, thread_index, device)
        else:
            GameACFFNetwork.use_mnih_2015 = self.use_mnih_2015
            self.local_network = GameACFFNetwork(self.action_size, thread_index, device)

        with tf.device(device):
            self.local_network.prepare_loss(entropy_beta=self.entropy_beta, critic_lr=0.5)
            local_vars = self.local_network.get_vars
            if self.finetune_upper_layers_only:
                local_vars = self.local_network.get_vars_upper
            var_refs = [v._ref() for v in local_vars()]

            self.gradients = tf.gradients(self.local_network.total_loss, var_refs)

        global_vars = global_network.get_vars
        if self.finetune_upper_layers_only:
            global_vars = global_network.get_vars_upper

        with tf.device(device):
            if self.clip_norm is not None:
                self.gradients, grad_norm = tf.clip_by_global_norm(self.gradients, self.clip_norm)
            self.gradients = list(zip(self.gradients, global_vars()))
            self.apply_gradients = grad_applier.apply_gradients(self.gradients)

            #self.apply_gradients = grad_applier.apply_gradients(
            #    global_vars(),
            #    self.gradients)

        self.sync = self.local_network.sync_from(
            global_network, upper_layers_only=self.finetune_upper_layers_only)

        self.game_state = GameState(
            env_id=self.env_id, display=False,
            no_op_max=30, human_demo=False, episode_life=True)

        self.local_t = 0

        self.initial_learning_rate = initial_learning_rate

        self.episode_reward = 0
        self.episode_steps = 0

        # variable controlling log output
        self.prev_local_t = 0

        self.is_demo_thread = False

        with tf.device(device):
            if self.load_demo_cam:
                self.action_meaning = self.game_state.env.unwrapped.get_action_meanings()
                self.local_network.build_grad_cam_grads()

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

    def choose_action(self, logits):
        """sample() in https://github.com/ppyht2/tf-a2c/blob/master/src/policy.py"""
        noise = np.random.uniform(0, 1, np.shape(logits))
        return np.argmax(logits - np.log(-np.log(noise)))

    def choose_action_with_high_confidence(self, pi_values, exclude_noop=True):
        actions_confidence = []
        # exclude NOOP action
        for action in range(1 if exclude_noop else 0, self.action_size):
            actions_confidence.append(pi_values[action][0][0])
        max_confidence_action = np.argmax(actions_confidence)
        confidence = actions_confidence[max_confidence_action]
        return (max_confidence_action+(1 if exclude_noop else 0)), confidence

    def set_summary_writer(self, writer):
        self.writer = writer

    def record_summary(self, score=0, steps=0, episodes=None, global_t=0, mode='Test'):
        summary = tf.Summary()
        summary.value.add(tag='{}/score'.format(mode), simple_value=float(score))
        summary.value.add(tag='{}/steps'.format(mode), simple_value=float(steps))
        if episodes is not None:
            summary.value.add(tag='{}/episodes'.format(mode), simple_value=float(episodes))
        self.writer.add_summary(summary, global_t)
        self.writer.flush()

    def set_start_time(self, start_time):
        self.start_time = start_time

    def generate_cam(self, sess, test_cam_si, global_t):
        action_array = []
        cam = []
        img = []

        cam_plus_img = []
        cam_side_img = []
        for i in range(len(test_cam_si)):
            # get max action per demo state
            readout_t = self.local_network.run_policy(sess, test_cam_si[i])
            action = np.argmax(readout_t)
            action_array.append(action)

            # convert action to one-hot vector
            action_onehot = [0.] * self.game_state.env.action_space.n
            action_onehot[action] = 1

            # compute grad cam for conv layer 3
            conv_value, conv_grad = self.local_network.evaluate_grad_cam(
                sess, test_cam_si[i], action_onehot)
            cam_img = visualize_cam(conv_value, conv_grad)

            overlay, side_by_side = generate_image_for_cam_video(
                test_cam_si[i],
                cam_img, global_t, i,
                self.action_meaning[action])

            cam_plus_img.append(overlay)
            cam_side_img.append(side_by_side)

        return cam_plus_img, cam_side_img

    def generate_cam_video(self, sess, time_per_step, global_t, folder, demo_memory_cam):
        # use one demonstration data to record cam
        # only need to make movie for demo data once
        cam_plus_img, cam_side_img = self.generate_cam(sess, demo_memory_cam, global_t)

        make_movie(
            cam_plus_img,
            folder + '/frames/demo-cam_plus_img{ep:010d}'.format(ep=(global_t)),
            duration=len(cam_plus_img)*time_per_step,
            true_image=True,
            salience=False)
        make_movie(
            cam_side_img,
            folder + '/frames/demo-cam_side_img{ep:010d}'.format(ep=(global_t)),
            duration=len(cam_side_img)*time_per_step,
            true_image=True,
            salience=False)
        del cam_plus_img, cam_side_img

    def testing(self, sess, max_steps, global_t, folder, demo_memory_cam=None):
        logger.info("Evaluate policy at global_t={}...".format(global_t))
        # copy weights from shared to local
        sess.run(self.sync)

        if self.load_demo_cam:
            self.generate_cam_video(sess, 0.03, global_t, folder, demo_memory_cam)

        episode_buffer = []
        self.game_state.reset(hard_reset=True)
        episode_buffer.append(self.game_state.get_screen_rgb())

        total_reward = 0
        total_steps = 0
        episode_reward = 0
        episode_steps = 0
        n_episodes = 0
        while max_steps > 0:
            #pi_ = self.local_network.run_policy(sess, self.game_state.s_t)
            pi_, value_, logits_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
            if False:
                action = np.random.choice(range(self.action_size), p=pi_)
            else:
                action = self.choose_action(logits_)

            if self.use_pretrained_model_as_advice:
                psi = self.psi if self.psi > 0.001 else 0.0
                if psi > np.random.rand():
                    model_pi = self.pretrained_model.run_policy(self.pretrained_model_sess, self.game_state.s_t)
                    model_action, confidence = self.choose_action_with_high_confidence(model_pi, exclude_noop=False)
                    if model_action > self.shaping_actions and confidence >= self.advice_confidence:
                        action = model_action

            # take action
            self.game_state.step(action)
            terminal = self.game_state.terminal

            if n_episodes == 0 and global_t % 5000000 == 0:
                episode_buffer.append(self.game_state.get_screen_rgb())

            episode_reward += self.game_state.reward
            episode_steps += 1
            max_steps -= 1

            # s_t = s_t1
            self.game_state.update()

            if terminal:
                if get_wrapper_by_name(self.game_state.env, 'EpisodicLifeEnv').was_real_done:
                    if n_episodes == 0 and global_t % 5000000 == 0:
                        time_per_step = 0.0167
                        images = np.array(episode_buffer)
                        make_movie(
                            images, folder + '/frames/image{ep:010d}'.format(ep=global_t),
                            duration=len(images)*time_per_step,
                            true_image=True, salience=False)
                        episode_buffer = []
                    n_episodes += 1
                    score_str = colored("score={}".format(episode_reward), "magenta")
                    steps_str = colored("steps={}".format(episode_steps), "blue")
                    log_data = (global_t, self.thread_index, n_episodes, score_str, steps_str, total_steps)
                    logger.debug("test: global_t={} worker={} trial={} {} {} total_steps={}".format(*log_data))
                    total_reward += episode_reward
                    total_steps += episode_steps
                    episode_reward = 0
                    episode_steps = 0

                self.game_state.reset(hard_reset=False)
                if self.use_lstm:
                    self.local_network.reset_state()

        if n_episodes == 0:
            total_reward = episode_reward
            total_steps = episode_steps
        else:
            # (timestep, total sum of rewards, total # of steps before terminating)
            total_reward = total_reward / n_episodes
            total_steps = total_steps // n_episodes

        log_data = (global_t, self.thread_index, total_reward, total_steps, n_episodes)
        logger.info("test: global_t={} worker={} final score={} final steps={} # trials={}".format(*log_data))

        self.record_summary(
            score=total_reward, steps=total_steps,
            episodes=n_episodes, global_t=global_t, mode='Test')

        # reset variables used in training
        self.episode_reward = 0
        self.episode_steps = 0
        self.game_state.reset(hard_reset=True)
        self.last_rho = 0.
        if self.is_demo_thread:
            self.replay_mem_reset()

        if self.use_lstm:
            self.local_network.reset_state()
        return total_reward, total_steps, n_episodes

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
        logger.debug("worker={} mem_reset demo_memory_idx={} demo_memory_start={}".format(self.thread_index, self.demo_memory_idx, self.demo_memory_count))
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
                log_msg = "lg={}".format(np.array_str(logits_, precision=4, suppress_small=True))
                log_msg += " pi={}".format(np.array_str(pi_, precision=4, suppress_small=True))
                log_msg += " V={:.4f}".format(value_)
                logger.debug(log_msg)

            # process replay memory
            self.replay_mem_process()

            # receive replay memory result
            reward = self.demo_memory_reward
            terminal = self.demo_memory_terminal

            self.episode_reward += reward

            if self.reward_type == 'LOG':
                reward = np.sign(reward) * np.log(1 + np.abs(reward))
            elif self.reward_type == 'CLIP':
                # clip reward
                reward = np.sign(reward)

            rewards.append(reward)

            self.local_t += 1
            self.episode_steps += 1

            # demo_memory_s_t1 -> demo_memory_s_t
            self.replay_mem_update()
            s_t = self.demo_memory_s_t

            if terminal or self.demo_memory_count == len(self.demo_memory[self.demo_memory_idx]):
                logger.debug("worker={} score={}".format(self.thread_index, self.episode_reward))
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

        cumulative_reward = 0.0
        if not terminal_end:
            cumulative_reward = self.local_network.run_value(sess, s_t)

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_state = []
        batch_action = []
        batch_adv = []
        batch_cumulative_reward = []

        # compute and accmulate gradients
        for(ai, ri, si, vi) in zip(actions, rewards, states, values):
            cumulative_reward = ri + self.gamma * cumulative_reward
            advantage = cumulative_reward - vi

            # convert action to one-hot vector
            a = np.zeros([self.action_size])
            a[ai] = 1

            batch_state.append(si)
            batch_action.append(a)
            batch_adv.append(advantage)
            batch_cumulative_reward.append(cumulative_reward)

        cur_learning_rate = self._anneal_learning_rate(global_t) #* 0.005

        if self.use_lstm:
            batch_state.reverse()
            batch_action.reverse()
            batch_adv.reverse()
            batch_cumulative_reward.reverse()

            sess.run(self.apply_gradients,
                     feed_dict = {
                         self.local_network.s: batch_state,
                         self.local_network.a: batch_action,
                         self.local_network.advantage: batch_adv,
                         self.local_network.cumulative_reward: batch_cumulative_reward,
                         self.local_network.initial_lstm_state: start_lstm_state,
                         self.local_network.step_size : [len(batch_action)],
                         self.learning_rate_input: cur_learning_rate} )

            # some demo episodes doesn't reach terminal state
            if reset_lstm_state:
                self.local_network.reset_state()
                reset_lstm_state = False
        else:
            sess.run(self.apply_gradients,
                     feed_dict = {
                         self.local_network.s: batch_state,
                         self.local_network.a: batch_action,
                         self.local_network.advantage: batch_adv,
                         self.local_network.cumulative_reward: batch_R,
                         self.learning_rate_input: cur_learning_rate} )

        if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= self.performance_log_interval):
            self.prev_local_t += self.performance_log_interval

        # return advancd local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t, demo_ended

    def process(self, sess, global_t, train_rewards):
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
            action = self.choose_action(logits_)

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
                log_msg1 = "lg={}".format(np.array_str(logits_, precision=4, suppress_small=True))
                log_msg2 = "pi={}".format(np.array_str(pi_, precision=4, suppress_small=True))
                log_msg3 = "V={:.4f}".format(value_)
                if self.use_pretrained_model_as_advice:
                    log_msg3 += " psi={:.4f}".format(self.psi)
                logger.debug(log_msg1)
                logger.debug(log_msg2)
                logger.debug(log_msg3)

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

            if self.reward_type == 'LOG':
                reward = np.sign(reward) * np.log(1 + np.abs(reward))
            elif self.reward_type == 'CLIP':
                # clip reward
                reward = np.sign(reward)

            rewards.append(reward)

            self.local_t += 1
            self.episode_steps += 1
            global_t += 1

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
                    self.record_summary(
                        score=self.episode_reward, steps=self.episode_steps,
                        episodes=None, global_t=global_t, mode='Train')
                    self.episode_reward = 0
                    self.episode_steps = 0
                    terminal_end = True

                self.last_rho = 0.
                if self.use_lstm:
                    self.local_network.reset_state()
                self.game_state.reset(hard_reset=False)
                break

        cumulative_reward = 0.0
        if not terminal:
            cumulative_reward = self.local_network.run_value(sess, self.game_state.s_t)

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_state = []
        batch_action = []
        batch_adv = []
        batch_cumulative_reward = []

        if self.use_pretrained_model_as_reward_shaping:
            rho.reverse()
            rho.append(self.last_rho)
            self.last_rho = rho[0]
            i = 0
            # compute and accumulate gradients
            for(ai, ri, si, vi) in zip(actions, rewards, states, values):
                # Wiewiora et al.(2003) Principled Methods for Advising RL agents
                # Look-Back Advice
                #F = rho[i] - (self.shaping_gamma**-1) * rho[i+1]
                #F = rho[i] - self.shaping_gamma * rho[i+1]
                f = (self.shaping_gamma**-1) * rho[i] - rho[i+1]
                if (i == 0 and terminal) or (f != 0 and (ri > 0 or ri < 0)):
                    #logger.warn("averted additional F in absorbing state")
                    F = 0.
                # if (F < 0. and ri > 0) or (F > 0. and ri < 0):
                #     logger.warn("Negative reward shaping F={} ri={} rho[s]={} rhos[s-1]={}".format(F, ri, rho[i], rho[i+1]))
                #     F = 0.
                cumulative_reward = (ri + f*self.shaping_factor) + self.gamma * cumulative_reward
                advantage = cumulative_reward - vi

                a = np.zeros([self.action_size])
                a[ai] = 1

                batch_state.append(si)
                batch_action.append(a)
                batch_adv.append(advantage)
                batch_cumulative_reward.append(cumulative_reward)
                i += 1
        else:
            def h(z, eps=10**-2):
                return (np.sign(z) * (np.sqrt(np.abs(z) + 1.) - 1.)) + (eps * z)

            def h_inv(z, eps=10**-2):
                return np.sign(z) * (np.square((np.sqrt(1 + 4 * eps * (np.abs(z) + 1 + eps)) - 1) / (2 * eps)) - 1)

            def h_log(z, eps=.6):
                return (np.sign(z) * np.log(1. + np.abs(z)) * eps)

            def h_inv_log(z, eps=.6):
                return np.sign(z) * (np.exp(np.abs(z) / eps) - 1)

            # compute and accumulate gradients
            for(ai, ri, si, vi) in zip(actions, rewards, states, values):
                if self.transformed_bellman:
                    cumulative_reward = h(ri + self.gamma * h_inv(cumulative_reward))
                else:
                    cumulative_reward = ri + self.gamma * cumulative_reward
                advantage = cumulative_reward - vi

                # convert action to one-hot vector
                a = np.zeros([self.action_size])
                a[ai] = 1

                batch_state.append(si)
                batch_action.append(a)
                batch_adv.append(advantage)
                batch_cumulative_reward.append(cumulative_reward)

        cur_learning_rate = self._anneal_learning_rate(global_t)

        if self.use_lstm:
            batch_state.reverse()
            batch_action.reverse()
            batch_adv.reverse()
            batch_cumulative_reward.reverse()

            sess.run(self.apply_gradients,
                feed_dict = {
                    self.local_network.s: batch_state,
                    self.local_network.a: batch_action,
                    self.local_network.advantage: batch_adv,
                    self.local_network.cumulative_reward: batch_cumulative_reward,
                    self.local_network.initial_lstm_state: start_lstm_state,
                    self.local_network.step_size : [len(batch_action)],
                    self.learning_rate_input: cur_learning_rate})
        else:
            sess.run(self.apply_gradients,
                feed_dict = {
                    self.local_network.s: batch_state,
                    self.local_network.a: batch_action,
                    self.local_network.advantage: batch_adv,
                    self.local_network.cumulative_reward: batch_cumulative_reward,
                    self.learning_rate_input: cur_learning_rate})

        if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= self.performance_log_interval):
            self.prev_local_t += self.performance_log_interval
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            logger.info("Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t, terminal_end
