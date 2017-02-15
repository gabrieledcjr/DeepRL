#!/usr/bin/env python
import tensorflow as tf
import cv2
import sys
import random
import numpy as np
from data_set import DataSet
from dqn_net_bn import DqnNetBn
from util import egreedy, get_action_index
import tables
import time

try:
    import cPickle as pickle
except ImportError:
    import pickle

class Experiment(object):
    def __init__(
        self, sess, network, game, resized_height, resized_width, phi_length, actions, batch,
        name, gamma, observe, explore, final_epsilon, init_epsilon, replay_memory,
        update_freq, save_freq, eval_freq, eval_max_steps, copy_freq,
        optimizer, learning_rate, epsilon, decay, momentum, tau,
        verbose, path, folder, slow, load_human_memory=False, train_max_steps=sys.maxint):
        """ Initialize experiment """
        self.sess = sess
        self.observe = observe
        self.explore = explore
        self.final_epsilon = final_epsilon
        self.init_epsilon = init_epsilon
        self.update_freq = update_freq # backpropagate frequency
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.eval_max_steps = eval_max_steps
        self.copy_freq = copy_freq # copy q to t-network frequency
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.phi_length = phi_length
        self.actions = actions
        self.batch = batch
        self.name = name
        self.path = path
        self.folder = folder
        self.load_human_memory = load_human_memory
        self.train_max_steps = train_max_steps

        if False: # Deterministic
            rng = np.random.RandomState(123456)
        else:
            rng = np.random.RandomState()
        self.D = DataSet(self.resized_width, self.resized_height, rng, replay_memory, self.phi_length, self.actions)
        self.net = network
        self.game_state = game.GameState()

    def _reset(self):
        do_nothing = np.zeros(self.actions)
        do_nothing[0] = 1
        observation, r_0, terminal = self.game_state.frame_step(0)

        observation = cv2.cvtColor(cv2.resize(observation, (self.resized_height, self.resized_width)), cv2.COLOR_BGR2GRAY)
        observation_t = observation / 255.0
        s_t = [ observation_t for _ in range(self.phi_length)]
        s_t = np.stack(tuple(s_t), axis = 2)
        return observation, s_t

    def _add_human_experiences(self):
        data = pickle.load(open(self.name + '_human_samples/' + self.name + '-dqn-all.pkl', 'rb'))
        terminals = data['D.terminal']
        actions = data['D.actions']
        rewards = data['D.rewards']
        h5file = tables.openFile(self.name + '_human_samples/' + self.name + '-dqn-images-all.h5', mode='r')
        imgs = h5file.root.images[:]
        h5file.close()
        print "\tMemory size={}".format(self.D.size)
        print "\tAdding {} human experiences...".format(data['D.size'])
        for i in range(data['D.size']):
            s = imgs[i]
            a = actions[i]
            r = rewards[i]
            t = terminals[i]
            self.D.add_sample(s, a, r, t)
        print "\tMemory size={}".format(self.D.size)

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
            h5file = tables.openFile(self.folder + '/' + self.name + '-dqn-images.h5', mode='r')
            self.D.imgs = h5file.root.images[:]
            h5file.close()
        else:
            print "Could not find old network weights"
            if self.load_human_memory:
                self._add_human_experiences()
            t = 0
            epsilon = self.init_epsilon
            rewards = {'train':[], 'eval':[]}
        return t, epsilon, rewards

    def test(self, show_gui=False):
        # re-initialize game for evaluation
        self.game_state.reinit(random_restart=False, is_testing=True)
        last_img, s_t = self._reset()
        max_steps = self.eval_max_steps
        total_reward = 0.0
        total_steps = 0
        sub_total_reward = 0.0
        sub_steps = 0
        n_episodes = 0
        time.sleep(0.5)
        while max_steps > 0:
            readout_t = self.net.evaluate([s_t])[0]
            action_index = get_action_index(readout_t, is_random=(random.random() <= 0.05), n_actions=self.actions)
            a_t = np.zeros([self.actions])
            a_t[action_index] = 1
            observation, r_t, terminal = self.game_state.frame_step(action_index, gui=show_gui)
            observation = cv2.cvtColor(cv2.resize(observation, (self.resized_height, self.resized_width)), cv2.COLOR_BGR2GRAY)
            observation_t = observation / 255.0
            observation_t = np.reshape(observation_t, (self.resized_height, self.resized_width, 1))
            s_t1 = np.append(observation_t, s_t[:,:,0:3], axis = 2)
            s_t = s_t1
            sub_total_reward += r_t
            sub_steps += 1
            max_steps -= 1
            if terminal:
                show_gui = False
                n_episodes += 1
                print "\tTRIAL", n_episodes, "/ REWARD", sub_total_reward, "/ STEPS", sub_steps, "/ TOTAL STEPS", total_steps
                self.game_state.reinit(random_restart=True, is_testing=True)
                _, s_t = self._reset()
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
        last_img, s_t = self._reset()
        self.t, self.epsilon, self.rewards = self._load()

        print "D size: ", self.D.size
        total_reward = 0.0
        sub_steps = 0

        while (self.t - self.observe) < self.train_max_steps:
            # Evaluation of policy
            if (self.t - self.observe) >= 0 and (self.t - self.observe) % self.eval_freq == 0:
                terminal = 0
                total_reward, total_steps, n_episodes = self.test()
                print "TIMESTEP", (self.t - self.observe), "/ AVE REWARD", total_reward, "/ AVE TOTAL STEPS", total_steps, "/ # EPISODES", n_episodes
                # re-initialize game for training
                self.game_state.reinit(random_restart=True)
                last_img, s_t = self._reset()
                sub_steps = 0
                time.sleep(0.5)

            # choose an action epsilon greedily
            readout_t = self.net.evaluate([s_t])[0]
            action_index = get_action_index(
                readout_t,
                is_random=(random.random() <= self.epsilon or self.t <= self.observe),
                n_actions=self.actions)
            a_t = np.zeros([self.actions])
            a_t[action_index] = 1

            # scale down epsilon
            if self.epsilon > self.final_epsilon and self.t > self.observe:
                self.epsilon -= (self.init_epsilon - self.final_epsilon) / self.explore

            # Training
            # run the selected action and observe next state and reward
            observation, r_t, terminal = self.game_state.frame_step(action_index, random_restart=True)
            terminal = True if (terminal or r_t == -1) else False
            sub_steps += 1
            observation = cv2.cvtColor(cv2.resize(observation, (self.resized_height, self.resized_width)), cv2.COLOR_BGR2GRAY)

            # store the transition in D
            terminal_ = terminal or ((self.t+1 - self.observe) >= 0 and (self.t+1 - self.observe) % self.eval_freq == 0)
            self.D.add_sample(last_img, a_t, r_t, (1 if terminal_ else 0))
            last_img = observation

            observation_t = observation / 255.0
            observation_t = np.reshape(observation_t, (self.resized_height, self.resized_width, 1))
            s_t1 = np.append(observation_t, s_t[:,:,0:3], axis = 2)

            # only train if done observing
            if self.t > self.observe and self.t % self.update_freq == 0:
                s_j_batch, a_batch, r_batch, s_j1_batch, terminals = self.D.random_batch(self.batch)
                # perform gradient step
                summary = self.net.train(s_j_batch, a_batch, r_batch, s_j1_batch, terminals, total_reward)
                self.net.add_summary(summary, self.t-self.observe)

                self.rewards['train'].append(round(r_t, 4))

            # update the old values
            s_t = s_t1
            self.t += 1

            if terminal:
                last_img, s_t = self._reset()
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
                pickle.dump(data, open(self.folder + '/' + self.name + '-dqn.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.rewards, open(self.folder + '/' + self.name + '-dqn-rewards.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                h5file = tables.openFile(self.folder + '/' + self.name + '-dqn-images.h5', mode='w', title='Images Array')
                root = h5file.root
                h5file.createArray(root, "images", self.D.imgs)
                h5file.close()

            # print info
            state = ""
            if self.t <= self.observe:
                state = "observe"
            elif self.t > self.observe and self.t <= self.observe + self.explore:
                state = "explore"
            else:
                state = "train"

            if self.t%1000 == 0:
                print "TIMESTEP", self.t, "/ STATE", state, "/ EPSILON", round(self.epsilon,4), "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t)

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
