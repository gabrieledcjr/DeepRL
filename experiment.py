#!/usr/bin/env python
import tensorflow as tf
import cv2
import sys
import os
import random
import numpy as np
import time

from termcolor import colored
from data_set import DataSet
from util import egreedy, get_action_index, make_gif, process_frame, get_compressed_images, save_compressed_images

try:
    import cPickle as pickle
except ImportError:
    import pickle

class Experiment(object):
    def __init__(
        self, sess, network, game_state, resized_height, resized_width, phi_length, batch,
        name, gamma, observe, explore, final_epsilon, init_epsilon, replay_memory,
        update_freq, save_freq, eval_freq, eval_max_steps, copy_freq,
        path, folder, load_human_memory=False, train_max_steps=sys.maxsize):
        """ Initialize experiment """
        self.sess = sess
        self.net = network
        self.game_state = game_state
        self.observe = observe
        self.explore = explore
        self.final_epsilon = final_epsilon
        self.init_epsilon = init_epsilon
        self.update_freq = update_freq # backpropagate frequency
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.eval_max_steps = eval_max_steps
        self.copy_freq = copy_freq # copy q to t-network frequency
        self.resized_h = resized_height
        self.resized_w = resized_width
        self.phi_length = phi_length
        self.batch = batch
        self.name = name
        self.path = path
        self.folder = folder
        self.load_human_memory = load_human_memory
        self.train_max_steps = train_max_steps
        self.wall_t = 0.0

        self.state_input = np.zeros((1, 84, 84, self.phi_length), dtype=np.uint8)
        self.D = replay_memory

        if not os.path.exists(self.folder + '/frames'):
            os.makedirs(self.folder + '/frames')

    def _reset(self, testing=False):
        self.state_input.fill(0)
        observation, r_0, terminal = self.game_state.frame_step(0)
        observation = process_frame(observation, self.resized_h, self.resized_w)
        if not testing:
            for _ in range(self.phi_length-1):
                empty_img = np.zeros((self.resized_w, self.resized_h), dtype=np.uint8)
                self.D.add_sample(empty_img, 0, 0, 0)
        return observation

    def _update_state_input(self, observation):
        self.state_input = np.roll(self.state_input, -1, axis=3)
        self.state_input[0, :, :, -1] = observation

    def _add_human_experiences(self):
        if self.name == 'pong' or self.name == 'breakout':
            # data were pickled using Python 2 which have compatibility issues in Python 3
            data = pickle.load(open(self.name + '_human_samples/' + self.name + '-dqn-all.pkl', 'rb'), encoding='latin1')
        else:
            data = pickle.load(open(self.name + '_human_samples/' + self.name + '-dqn-all.pkl', 'rb'))
        terminals = data['D.terminal']
        actions = data['D.actions']
        rewards = data['D.rewards']
        imgs = get_compressed_images(self.name + '_human_samples/' + self.name + '-dqn-images-all.h5' + '.gz')
        print ("\tMemory size={}".format(self.D.size))
        print ("\tAdding {} human experiences...".format(data['D.size']))
        for i in range(data['D.size']):
            s = imgs[i]
            a = actions[i]
            r = rewards[i]
            t = terminals[i]
            self.D.add_sample(s, a, r, t)
        print ("\tMemory size={}".format(self.D.size))
        time.sleep(2)

    def _load(self):
        if self.net.load():
            rewards = pickle.load(open(self.folder + '/' + self.name + '-dqn-rewards.pkl', 'rb'))
            data = pickle.load(open(self.folder + '/' + self.name + '-dqn.pkl', 'rb'))
            self.D.width = data['D.width']
            self.D.height = data['D.height']
            self.D.max_steps = data['D.max_steps']
            self.D.phi_length = data['D.phi_length']
            self.D.num_actions = data['D.num_actions']
            self.D.actions = data['D.actions']
            self.D.rewards = data['D.rewards']
            self.D.terminal = data['D.terminal']
            self.D.bottom = data['D.bottom']
            self.D.top = data['D.top']
            self.D.size = data['D.size']
            t = data['t']
            epsilon = data['epsilon']
            self.D.imgs = get_compressed_images(self.folder + '/' + self.name + '-dqn-images.h5' + '.gz')
        else:
            print ("Could not find old network weights")
            if self.load_human_memory:
                self._add_human_experiences()
            t = 0
            epsilon = self.init_epsilon
            rewards = {'train':[], 'eval':[]}
        return t, epsilon, rewards

    def test(self, render=False):
        # re-initialize game for evaluation
        episode_buffer = []
        self.game_state.reinit(random_restart=False, terminate_loss_of_life=False)
        observation = self._reset(testing=True)
        episode_buffer.append(self.game_state.screen_buffer)

        max_steps = self.eval_max_steps
        total_reward = 0.0
        total_steps = 0
        sub_total_reward = 0.0
        sub_steps = 0
        n_episodes = 0
        time.sleep(0.5)
        while max_steps > 0:
            self._update_state_input(observation)
            readout_t = self.net.evaluate(self.state_input)[0]
            action = get_action_index(readout_t, is_random=(random.random() <= 0.05), n_actions=self.game_state.n_actions)
            observation, reward, terminal = self.game_state.frame_step(action, render=render)
            if n_episodes == 0:
                episode_buffer.append(observation)
            observation = process_frame(observation, self.resized_h, self.resized_w)
            sub_total_reward += reward
            sub_steps += 1
            max_steps -= 1
            if terminal:
                if n_episodes == 0:
                    time_per_step = 0.05
                    images = np.array(episode_buffer)
                    make_gif(
                        images, self.folder + '/frames/image{ep:010d}.gif'.format(ep=(self.t-self.observe)),
                        duration=len(images)*time_per_step,
                        true_image=True, salience=False)
                    episode_buffer = []
                n_episodes += 1
                print ("\tTRIAL", n_episodes, "/ REWARD", sub_total_reward, "/ STEPS", sub_steps, "/ TOTAL STEPS", total_steps)
                self.game_state.reinit(random_restart=True, terminate_loss_of_life=False)
                observation = self._reset(testing=True)
                total_reward += sub_total_reward
                total_steps += sub_steps
                sub_total_reward = 0.0
                sub_steps = 0
                time.sleep(0.5)
        # (timestep, total sum of rewards, toal # of steps before terminating)
        total_reward = total_reward / max(1, n_episodes)
        total_steps = total_steps / max(1, n_episodes)
        total_reward = round(total_reward, 4)
        self.rewards['eval'].append(((self.t - self.observe), total_reward, total_steps))
        return total_reward, total_steps, n_episodes

    def run(self):
        # get the first state by doing nothing and preprocess the image to 80x80x4
        observation = self._reset()
        self.t, self.epsilon, self.rewards = self._load()

        # set start time
        self.start_time = time.time() - self.wall_t

        print ("D size: ", self.D.size)
        total_reward = 0.0
        sub_steps = 0

        while (self.t - self.observe) < self.train_max_steps:
            # Evaluation of policy
            if (self.t - self.observe) >= 0 and (self.t - self.observe) % self.eval_freq == 0:
                terminal = 0
                total_reward, total_steps, n_episodes = self.test()
                self.net.add_accuracy(total_reward, total_steps, n_episodes, (self.t - self.observe))
                print ("TIMESTEP", (self.t - self.observe), "/ AVE REWARD", total_reward, "/ AVE TOTAL STEPS", total_steps, "/ # EPISODES", n_episodes)
                # re-initialize game for training
                self.game_state.reinit(random_restart=True)
                observation = self._reset()
                sub_steps = 0
                time.sleep(0.5)

            # choose an action epsilon greedily
            self._update_state_input(observation)
            readout_t = self.net.evaluate(self.state_input)[0]
            action = get_action_index(
                readout_t,
                is_random=(random.random() <= self.epsilon or self.t <= self.observe),
                n_actions=self.game_state.n_actions)

            # scale down epsilon
            if self.epsilon > self.final_epsilon and self.t > self.observe:
                self.epsilon -= (self.init_epsilon - self.final_epsilon) / self.explore

            # Training
            # run the selected action and observe next state and reward
            next_observation, reward, terminal = self.game_state.frame_step(action, random_restart=True)
            next_observation = process_frame(next_observation, self.resized_h, self.resized_w)
            terminal_ = terminal or ((self.t+1 - self.observe) >= 0 and (self.t+1 - self.observe) % self.eval_freq == 0)

            # store the transition in D
            self.D.add_sample(observation, action, reward, (1 if terminal_ else 0))

            # only train if done observing
            if self.t > self.observe and self.t % self.update_freq == 0:
                s_j_batch, a_batch, r_batch, s_j1_batch, terminals = self.D.random_batch(self.batch)
                # perform gradient step
                summary = self.net.train(s_j_batch, a_batch, r_batch, s_j1_batch, terminals)
                self.net.add_summary(summary, self.t-self.observe)

                self.rewards['train'].append(round(reward, 4))

            # update the old values
            sub_steps += 1
            self.t += 1
            observation = next_observation

            if terminal:
                observation = self._reset()
                sub_steps = 0

            # save progress every SAVE_FREQ iterations
            if (self.t-self.observe) % self.save_freq == 0:
                self.net.save(self.t)

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
                        'D.size':self.D.size,
                        'epsilon':self.epsilon,
                        't':self.t}
                print (colored('Saving data...', 'blue'))
                pickle.dump(data, open(self.folder + '/' + self.name + '-dqn.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.rewards, open(self.folder + '/' + self.name + '-dqn-rewards.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                print (colored('Successfully saved data!', 'green'))
                print (colored('Compressing and saving replay memory...', 'blue'))
                save_compressed_images(self.folder + '/' + self.name + '-dqn-images.h5', self.D.imgs)
                print (colored('Compressed and saved replay memory', 'green'))

                # write wall time
                self.wall_t = time.time() - self.start_time
                print ('Total time: {} seconds'.format(self.wall_t))

            # print info
            state = ""
            if self.t <= self.observe:
                state = "observe"
            elif self.t > self.observe and self.t <= self.observe + self.explore:
                state = "explore"
            else:
                state = "train"

            if self.t%1000 == 0:
                print ("TIMESTEP", self.t, "/ STATE", state, "/ EPSILON", round(self.epsilon,4), "/ ACTION", action, "/ REWARD", reward, "/ Q_MAX %e" % np.max(readout_t))

NUM_THREADS = 16
def playGame():
    #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, intra_op_parallelism_threads=NUM_THREADS)) as sess:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True, intra_op_parallelism_threads=NUM_THREADS)) as sess:
        with tf.device('/gpu:'+os.environ["CUDA_VISIBLE_DEVICES"]):
            train(sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
