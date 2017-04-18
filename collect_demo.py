# -*- coding: utf-8 -*-
import numpy as np
import time
import random
from datetime import datetime
from termcolor import colored
from util import prepare_dir, process_frame, save_compressed_images, get_action_index

try:
    import cPickle as pickle
except ImportError:
    import pickle


class CollectDemonstration(object):

    def __init__(
        self, game_state, resized_height, resized_width, phi_length, name,
        replay_memory, terminate_loss_of_life=False, folder='', sample_num=0):
        """ Initialize collection of demo """
        assert sample_num > 0
        self.file_num = sample_num
        self.game_state = game_state
        self.resized_h = resized_height
        self.resized_w = resized_width
        self.phi_length = phi_length
        self.name = name
        self.D = replay_memory
        self.terminate_loss_of_life = terminate_loss_of_life

        self._skip = 1
        if self.game_state._env.frameskip == 1:
            self._skip = 4

        self.state_input = np.zeros(
            (1, self.resized_h, self.resized_w, self.phi_length),
            dtype=np.uint8)
        self.folder = folder + '/{n:03d}/'.format(name=self.name, n=self.file_num)
        prepare_dir(self.folder, empty=True)

    def _reset(self):
        self.state_input.fill(0)
        observation, r_0, terminal = self.game_state.frame_step(0, render=True)
        observation = process_frame(observation, self.resized_h, self.resized_w)
        for _ in range(self.phi_length-1):
            empty_img = np.zeros((self.resized_w, self.resized_h), dtype=np.uint8)
            self.D.add_sample(empty_img, 0, 0, 0)
        return observation

    def _update_state_input(self, observation):
        self.state_input = np.roll(self.state_input, -1, axis=3)
        self.state_input[0, :, :, -1] = observation

    def run(self, minutes_limit=5, demo_type=0, model_net=None):
        imgs = []
        acts = []
        rews = []
        terms = []

        rewards = {'train':[], 'eval':[]}

        # regular game
        start_time = datetime.now()
        timeout_start = time.time()
        timeout = 60 * minutes_limit
        t = 0
        terminal = False
        terminal_force = False
        is_reset = True
        total_reward = 0.0
        score1 = score2 = 0
        sub_t = 0
        sub_r = 0.
        rewards = []
        sub_steps = []
        total_episodes = 0

        # re-initialize game for evaluation
        self.game_state.reinit(
            render=True, random_restart=True,
            terminate_loss_of_life=self.terminate_loss_of_life)
        observation = self._reset()

        while True:
            if demo_type == 1: # RANDOM AGENT
                action = np.random.randint(self.game_state.n_actions)
            elif demo_type == 2: # MODEL AGENT
                if sub_t % self._skip == 0:
                    self._update_state_input(observation)
                    readout_t = model_net.evaluate(self.state_input)[0]
                    action = get_action_index(readout_t, is_random=False, n_actions=self.game_state.n_actions)
            else: # HUMAN
                action = self.game_state.human_agent_action

            next_observation, reward, terminal = self.game_state.frame_step(action, render=True, random_restart=True)
            next_observation = process_frame(next_observation, self.resized_h, self.resized_w)
            terminal = True if terminal or (time.time() > timeout_start + timeout) else False

            # store the transition in D
            # when using frameskip=1, should store every four steps
            if sub_t % self._skip == 0:
                self.D.add_sample(observation, action, reward, terminal)
            observation = next_observation
            sub_r += reward
            total_reward += reward

            #time.sleep(0.0166666)
            sub_t += 1
            t += 1

            # Ensure that D does not reach max memory that mitigate
            # problems when combining different human demo files
            if (self.D.size + 3) == self.D.max_steps:
                terminal_force = True
                terminal = True

            if terminal:
                total_episodes += 1
                rewards.append(sub_r)
                sub_steps.append(sub_t)
                sub_r = 0.
                sub_t = 0
                self.game_state.reinit(
                    render=True, random_restart=True,
                    terminate_loss_of_life=self.terminate_loss_of_life)
                observation = self._reset()
                is_reset = True
                time.sleep(0.5)

                if terminal_force or time.time() > timeout_start + timeout:
                    break

        if demo_type == 0: # HUMAN
            self.game_state.stop_thread = True

        print ("Duration: {}".format(datetime.now() - start_time))
        print ("Total # of episodes: {}".format(total_episodes))
        print ("Mean steps: {} / Mean reward: {}".format(t/total_episodes,total_reward/total_episodes))
        print ("\tsteps / episode:", sub_steps)
        print ("\treward / episode:", rewards)
        print ("Total Replay memory saved: {}".format(self.D.size))

        # Resize replay memory to exact memory size
        self.D.resize()
        data = {'D.width':self.D.width,
                'D.height':self.D.height,
                'D.max_steps':self.D.max_steps,
                'D.phi_length':self.D.phi_length,
                'D.num_actions':self.D.num_actions,
                'D.actions':self.D.actions,
                'D.rewards':self.D.rewards,
                'D.terminal':self.D.terminal,
                'D.bottom':self.D.bottom,
                'D.top':self.D.top,
                'D.size':self.D.size}
        images = self.D.imgs
        pkl_file = '{name}-dqn.pkl'.format(name=self.name)
        h5_file = '{name}-dqn-images.h5'.format(name=self.name)
        pickle.dump(data, open(self.folder + pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)
        print (colored('Compressing and saving replay memory...', 'blue'))
        save_compressed_images(self.folder + h5_file, images)
        print (colored('Compressed and saved replay memory', 'green'))
