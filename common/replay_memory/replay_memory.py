#!/usr/bin/env python3
"""
Copyright (c) 2014, Nathan Sprague
All rights reserved.

Original code: https://goo.gl/dp2qRV

This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.
"""

import numpy as np
import time
import coloredlogs, logging
import random

from collections import defaultdict
from common.util import save_compressed_images, get_compressed_images

try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger("replay_memory")

class ReplayMemory(object):
    """
    This replay memory assumes it's a single episode of memory in sequence
    """
    def __init__(self,
        width=1, height=1, rng=np.random.RandomState(),
        max_steps=10, phi_length=4, num_actions=1, wrap_memory=False,
        full_state_size=1013):
        """Construct a replay memory.

        Arguments:
            width, height - image size
            max_steps - the number of time steps to store
            phi_length - number of images to concatenate into a state
            rng - initialized numpy random number generator, used to
            choose random minibatches

        """
        # Store arguments.
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.phi_length = phi_length
        self.num_actions = num_actions
        self.rng = rng
        self.full_state_size = full_state_size

        # Allocate the circular buffers and indices.
        self.imgs = np.zeros((self.max_steps, height, width), dtype=np.uint8)
        self.actions = np.zeros(self.max_steps, dtype=np.uint8)
        self.rewards = np.zeros(self.max_steps, dtype=np.float32)
        self.terminal = np.zeros(self.max_steps, dtype=np.uint8)
        self.lives = np.zeros(self.max_steps, dtype=np.int32)
        self.full_state = np.zeros((self.max_steps, full_state_size), dtype=np.uint8)

        self.size = 0
        self.imgs_normalized = False

        self.wrap_memory = wrap_memory
        self.bottom = 0
        self.top = 0
        self.array_per_action = None

    def close(self):
        del self.imgs
        del self.actions
        del self.terminal
        del self.lives
        del self.full_state
        del self.array_per_action

    def normalize_images(self):
        if not self.imgs_normalized:
            logger.info("Normalizing images...")
            temp = self.imgs
            self.imgs = temp.astype(np.float32) / 255.0
            del temp
            self.imgs_normalized = True
            logger.info("Images normalized")

    def propagate_rewards(self, gamma=0.95, clip=False, normalize=False, minmax_scale=False, exclude_outlier=False, max_reward=0):
        logger.info("Propagating rewards...")
        logger.info("    reward size: {}".format(np.shape(self.rewards)[0]))
        logger.info("    gamma: {}".format(gamma))
        logger.info("    clip: {}".format(clip))
        logger.info("    normalize: {}".format(normalize))
        logger.info("    minmax_scale: {}".format(minmax_scale))

        logger.debug("    mean: {}".format(np.mean(np.abs(self.rewards))))
        logger.debug("    median: {}".format(np.median(np.abs(self.rewards))))

        if clip:
            np.clip(self.rewards, -1., 1., out=self.rewards)
        elif exclude_outlier and max_reward != 0:
            rewards = self.rewards[np.nonzero(self.rewards)]
            outliers = self.rewards[np.abs(self.rewards - np.mean(rewards)) > 2*np.std(rewards)]
            logger.debug("    outliers: {}".format(outliers))
            for outlier in outliers:
                if outlier != 0:
                    self.rewards[self.rewards == outlier] = max_reward if outlier > 0 else -max_reward
        if normalize and max_reward != 0:
            logger.debug("    max_reward: {}".format(max_reward))
            self.rewards = self.rewards / max_reward

        for i in range(self.size-2, 0, -1):
            #if self.rewards[i] != 0:
            self.rewards[i] = self.rewards[i] + gamma*self.rewards[i+1]

        if minmax_scale:
            from sklearn.preprocessing import MinMaxScaler
            rewards = self.rewards.reshape(-1, 1)
            scaler.fit(rewards)
            rewards = scaler.transform(rewards)
            self.rewards = rewards.reshape(-1)

        logger.debug("    max_reward: {}".format(np.linalg.norm(self.rewards, np.inf)))
        logger.debug("    min_reward: {}".format(np.min(np.abs(self.rewards[np.nonzero(self.rewards)]))))
        logger.info("Rewards propagated!")

    def resize(self):
        if self.max_steps == self.size:
            return
        logger.info("Resizing replay memory...")
        logger.debug("Current specs: size={} max_steps={}".format(self.size, self.max_steps))
        logger.debug("    images shape: {}".format(np.shape(self.imgs)))
        logger.debug("    actions shape: {}".format(np.shape(self.actions)))
        logger.debug("    rewards shape: {}".format(np.shape(self.rewards)))
        logger.debug("    terminal shape: {}".format(np.shape(self.terminal)))
        logger.debug("    lives shape: {}".format(np.shape(self.lives)))
        logger.debug("    full_state shape: {}".format(np.shape(self.full_state)))
        tmp_imgs = np.delete(self.imgs, range(self.size,self.max_steps), axis=0)
        tmp_actions = np.delete(self.actions, range(self.size, self.max_steps), axis=0)
        tmp_rewards = np.delete(self.rewards, range(self.size, self.max_steps), axis=0)
        tmp_terminal = np.delete(self.terminal, range(self.size, self.max_steps), axis=0)
        tmp_lives = np.delete(self.lives, range(self.size, self.max_steps), axis=0)
        tmp_fullstate = np.delete(self.full_state, range(self.size, self.max_steps), axis=0)
        del self.imgs, self.actions, self.rewards, self.terminal, \
            self.lives, self.full_state
        self.imgs = tmp_imgs
        self.actions = tmp_actions
        self.rewards = tmp_rewards
        self.terminal = tmp_terminal
        self.lives = tmp_lives
        self.full_state = tmp_fullstate
        self.max_steps = self.size
        logger.info("Resizing completed!")
        logger.debug("Updated specs: size={} max_steps={}".format(self.size, self.max_steps))
        logger.debug("    images shape: {}".format(np.shape(self.imgs)))
        logger.debug("    actions shape: {}".format(np.shape(self.actions)))
        logger.debug("    rewards shape: {}".format(np.shape(self.rewards)))
        logger.debug("    terminal shape: {}".format(np.shape(self.terminal)))
        logger.debug("    lives shape: {}".format(np.shape(self.lives)))
        logger.debug("    full_state shape: {}".format(np.shape(self.full_state)))

    def add(self, img, action, reward, terminal, lives, fullstate=None):
        """Add a time step record. Storing in replay memory should follow the following format:
        s0 = [img0, img1, img2, img3]
        s1 = [img1, img2, img3, img4]
        img0 | img1 | img2 | img3 | img4  <= image memory
                           |   s0 |   s1  <= state memory
                           |      |   a0  <= action memory
                           |      |   r1  <= reward memory
                           |      |   t1  <= terminal memory
                           |   l0 |   l1  <= lives memory
                           |  fs0 |  fs1  <= full state memory

        Arguments:
            img -- observed image
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
        assert fullstate is not None
        if not self.wrap_memory and self.size == self.max_steps:
            logger.warn("Memory is full. Data not added!")
            return

        if self.wrap_memory:
            idx = self.top
        else:
            idx = self.size

        self.imgs[idx] = img
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.terminal[idx] = terminal
        self.lives[idx] = lives
        self.full_state[idx] = fullstate

        if self.wrap_memory and self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1

        if self.wrap_memory:
            self.top = (self.top + 1) % self.max_steps

    def __len__(self):
        """Return an approximate count of stored state transitions."""
        return max(0, self.size - self.phi_length)

    def __getitem__(self, key):
        """Retrieving from replay memory follows the following format:
        if key = 0,
        s0 = [img0, img1, img2, img3]
        s1 = [img1, img2, img3, img4]
        img0 | img1 | img2 | img3 | img4  <= image memory
                           |   s0 |   s1  <= state memory
                           |      |   a0  <= action memory
                           |      |   r1  <= reward memory
                           |      |   t1  <= terminal memory
                           |   l0 |   l1  <= lives memory
                           |  fs0 |  fs1  <= full state memory
        """
        indices = np.arange(key, key + self.phi_length)
        indices_next = indices + 1
        end_index = key + self.phi_length - 1

        s0 = np.zeros(
            (self.height, self.width, self.phi_length),
            dtype=np.float32 if self.imgs_normalized else np.uint8)
        s1 = np.zeros(
            (self.height, self.width, self.phi_length),
            dtype=np.float32 if self.imgs_normalized else np.uint8)
        fs0 = np.zeros(self.full_state_size, np.uint8)

        if self.wrap_memory:
            mode = 'wrap'
            if np.any(self.terminal.take(indices, mode='wrap')):
                return None, None, None, None, None, None, None, None
        else:
            mode = 'raise'
            if end_index >= self.size or np.any(self.terminal.take(indices)):
                return None, None, None, None, None, None, None, None

        # s_t current state, action, lives, full state
        temp = self.imgs.take(indices, axis=0, mode=mode)
        for i in range(self.phi_length):
            s0[:, :, i] = temp[i]

        l0 = self.lives.take(end_index, mode=mode)
        fs0[:] = self.full_state.take(end_index, axis=0, mode=mode)

        # get action on next index even if it's for s0
        a0 = self.actions.take(end_index+1, mode=mode)

        # s_t+1 next state, reward, terminal, lives
        temp = self.imgs.take(indices_next, axis=0, mode=mode)
        for i in range(self.phi_length):
            s1[:, :, i] = temp[i]
        r1 = self.rewards.take(end_index+1, mode=mode)
        t1 = self.terminal.take(end_index+1, mode=mode)
        l1 = self.lives.take(end_index+1, mode=mode)

        return s0, a0, l0, fs0, s1, r1, t1, l1

    def __str__(self):
        specs = "Replay memory:\n"
        specs += "  size:{}\n".format(self.size)
        specs += "  max_steps:{}\n".format(self.max_steps)
        specs += "  imgs shape:{}\n".format(np.shape(self.imgs))
        return specs

    def get_item(self, key):
        return self.__getitem__(key)

    def sample_sequential(self, batch_size):
        """Return corresponding states, actions, rewards, terminal status, and
        next_states for batch_size randomly chosen state transitions.
        """
        assert not self.wrap_memory

        # Allocate the response.
        states = np.zeros(
            (batch_size, 84, 84, self.phi_length),
            dtype=np.float32 if self.imgs_normalized else np.uint8)
        actions = np.zeros((batch_size, self.num_actions), dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        terminals = np.zeros(batch_size, dtype=np.int)
        # lives = np.zeros(batch_size, dtype=np.int)

        # Randomly choose a time step from the replay memory
        # within requested batch_size
        # randint low (inclusive) to high (exclusive)
        high = (self.size + 1) - (self.phi_length + batch_size)
        assert high > 0 # crash if not enough memory

        # ensure no terminal besides the last index
        while True:
            self.random_index = self.rng.randint(0, high)
            indices = np.arange(self.random_index, self.random_index + batch_size + self.phi_length)
            if not np.any(self.terminal.take(indices[:-1])):
                index = self.random_index
                break

        for count in range(batch_size):
            s0, a0, l0, fs0, s1, r1, t1, l1 = self[index]
            # Add the state transition to the response.
            states[count] = np.copy(s0)
            actions[count][a0] = 1. # convert to one-hot vector
            rewards[count] = r1
            terminals[count] = t1
            index += 1

        return states, actions, rewards, terminals

    def create_index_array_per_action(self):
        assert not self.wrap_memory
        self.array_per_action = defaultdict(list)
        for index in range(len(self)):
            s0, a0, l0, fs0, s1, r1, t1, l1 = self[index]
            if s0 is None or s1 is None:
                continue
            self.array_per_action[a0].append(index)

    def sample_proportional(self, batch_size, batch_proportion, onevsall=False, n_class=None):
        """Return corresponding states, actions, rewards, terminal status, and
        next_states for batch_size randomly chosen state transitions.
        """
        assert not self.wrap_memory
        assert batch_size == sum(batch_proportion)

        if self.array_per_action is None:
            self.create_index_array_per_action()

        # Allocate the response.
        states = np.zeros((batch_size, self.height, self.width, self.phi_length), dtype=np.uint8)
        if onevsall:
            actions = np.zeros((batch_size, 2), dtype=np.float32)
        else:
            actions = np.zeros((batch_size, self.num_actions), dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        terminals = np.zeros(batch_size, dtype=np.int)

        count = 0
        for action, proportion in enumerate(batch_proportion):
            for _ in range(proportion):
                index = random.choice(self.array_per_action[action])
                s0, a0, l0, fs0, s1, r1, t1, l1 = self[index]

                # Add the state transition to the response.
                states[count] = np.copy(s0)
                if onevsall:
                    if a0 == n_class:
                        actions[count][0] = 1
                    else:
                        actions[count][1] = 1
                else:
                    actions[count][a0] = 1 # convert to one-hot vector
                rewards[count] = r1
                terminals[count] = t1
                count += 1

        return states, actions, rewards, terminals

    def sample2(self, batch_size, onevsall=False, n_class=None):
        """Return corresponding states, actions, rewards, terminal status, and
        next_states for batch_size randomly chosen state transitions.
        """
        assert not self.wrap_memory
        # Allocate the response.
        states = np.zeros((batch_size, self.height, self.width, self.phi_length), dtype=np.uint8)
        if onevsall:
            actions = np.zeros((batch_size, 2), dtype=np.float32)
        else:
            actions = np.zeros((batch_size, self.num_actions), dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        terminals = np.zeros(batch_size, dtype=np.int)
        # lives = np.zeros(batch_size, dtype=np.int)

        # Randomly choose a time step from the replay memory
        # within requested batch_size
        # randint low (inclusive) to high (exclusive)
        high = self.size - self.phi_length
        assert high > 0 # crash if not enough memory

        count = 0
        while count < batch_size:
            index = self.rng.randint(0, high)

            s0, a0, l0, fs0, s1, r1, t1, l1 = self[index]
            if s0 is None or s1 is None:
                continue
            # Add the state transition to the response.
            states[count] = np.copy(s0)
            if onevsall:
                if a0 == n_class:
                    actions[count][0] = 1
                else:
                    actions[count][1] = 1
            else:
                actions[count][a0] = 1 # convert to one-hot vector
            rewards[count] = r1
            terminals[count] = t1
            count += 1

        return states, actions, rewards, terminals

    def sample(self, batch_size, onevsall=False, n_class=None, reward_type=''):
        """Return corresponding states, actions, rewards, terminal status, and
        next_states for batch_size randomly chosen state transitions.
        reward_type = CLIP | LOG
        """
        assert self.wrap_memory
        # Allocate the response.
        states = np.zeros((batch_size, self.height, self.width, self.phi_length), dtype=np.uint8)
        next_states = np.zeros((batch_size, self.height, self.width, self.phi_length), dtype=np.uint8)
        if onevsall:
            actions = np.zeros((batch_size, 2), dtype=np.float32)
        else:
            actions = np.zeros((batch_size, self.num_actions), dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        terminals = np.zeros(batch_size, dtype=np.int)
        # lives = np.zeros(batch_size, dtype=np.int)

        high = self.bottom + self.size - self.phi_length
        assert high > 0 # crash if not enough memory

        count = 0
        while count < batch_size:
            index = self.rng.randint(self.bottom, high)
            indices = np.arange(index, index + self.phi_length + 1)
            end_index = index + self.phi_length - 1

            if np.any(self.terminal.take(indices[:-1], mode='wrap')):
                continue

            s0, a0, l0, fs0, s1, r1, t1, l1 = self[index]
            if s0 is None or s1 is None:
                continue
            # Add the state transition to the response.
            states[count] = np.copy(s0)
            next_states[count] = np.copy(s1)
            if onevsall:
                if a0 == n_class:
                    actions[count][0] = 1
                else:
                    actions[count][1] = 1
            else:
                actions[count][a0] = 1 # convert to one-hot vector
            if reward_type == 'CLIP':
                r1 = np.sign(r1)
            elif reward_type == 'LOG':
                r1 = np.sign(r1) * np.log(1. + np.abs(r1))
            rewards[count] = r1
            terminals[count] = t1
            count += 1

        return states, actions, rewards, terminals, next_states

    def save(self, name=None, folder=None, resize=False):
        assert name is not None
        assert folder is not None

        if resize:
            # Resize replay memory to exact memory size
            self.resize()
        data = {'width':self.width,
                'height':self.height,
                'max_steps':self.max_steps,
                'phi_length':self.phi_length,
                'num_actions':self.num_actions,
                'actions':self.actions,
                'rewards':self.rewards,
                'terminal':self.terminal,
                'lives':self.lives,
                'full_state_size': self.full_state_size,
                'full_state': self.full_state,
                'size':self.size,
                'wrap_memory':self.wrap_memory,
                'top':self.top,
                'bottom':self.bottom,
                'imgs_normalized':self.imgs_normalized}
        images = self.imgs
        pkl_file = '{}.pkl'.format(name)
        h5_file = '{}-images.h5'.format(name)
        pickle.dump(data, open(folder + '/' + pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)
        logger.info('Compressing and saving replay memory...')
        save_compressed_images(folder + '/' + h5_file, images)
        logger.info('Compressed and saved replay memory')

    def load(self, name=None, folder=None):
        assert name is not None
        assert folder is not None

        logger.info('Load memory from ' + folder + '...')
        pkl_file = '{}.pkl'.format(name)
        h5_file = '{}-images.h5'.format(name)
        data = pickle.load(open(folder + '/' + pkl_file, 'rb'))
        self.width = data['width']
        self.height = data['height']
        self.max_steps = data['max_steps']
        self.phi_length = data['phi_length']
        self.num_actions = data['num_actions']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.terminal = data['terminal']
        self.lives = data.get('lives', np.zeros(self.max_steps, dtype=np.int32))
        self.full_state_size = data.get('full_state_size', self.full_state_size)
        self.full_state = data.get('full_state', np.zeros((self.max_steps, self.full_state_size), dtype=np.uint8))
        self.size = data['size']
        self.wrap_memory = data['wrap_memory']
        self.top = data['top']
        self.bottom = data['bottom']
        self.imgs_normalized = data['imgs_normalized']
        self.imgs = get_compressed_images(folder + '/' + h5_file + '.gz')


def test_1(env_id):
    folder = "demo_samples/{}".format(env_id.replace('-', '_'))
    rm = ReplayMemory()
    rm.load(name=env_id, folder=(folder + '/001'))
    print (rm)

    for i in range(1000):
        states, actions, rewards, terminals = rm.sample2(20)

    import cv2
    count = 0
    state, _, _, _, _, _, _, _ = rm[count]
    print ("shape:", np.shape(state))
    while count < len(rm):
        state, _, _, _, _, _, _, _ = rm[count]
        cv2.imshow(env_id, state)
        cv2.imshow("one", state[:,:,0])
        cv2.imshow("two", state[:,:,1])
        cv2.imshow("three", state[:,:,2])
        cv2.imshow("four", state[:,:,3])
        # diff1 = cv2.absdiff(state[:,:,3], state[:,:,2])
        # diff2 = cv2.absdiff(state[:,:,3], state[:,:,1])
        # diff3 = cv2.absdiff(state[:,:,3], state[:,:,0])
        # diff = cv2.addWeighted(diff1, 0.8, diff2, 0.2, 0.0)
        # diff = cv2.addWeighted(diff, 0.8, diff3, 0.2, 0.0)
        # cv2.imshow("difference", diff)
        cv2.waitKey(20)
        count += 1
    print ("total transitions:", len(rm))
    print ("size:", rm.size)

def test_2(env_id):
    folder = "demo_samples/{}".format(env_id.replace('-', '_'))
    rm = ReplayMemory()
    rm.load(name=env_id, folder=(folder + '/001'))
    print(rm)

    print(len(rm))
    s0, a0, l0, fs0, s1, r1, t1, l1 = rm[len(rm)-1]
    print(s0)
    print(s1)
    print(a0, l0, r1, t1, l1)

    print()
    rm.normalize_images()
    s0, a0, l0, fs0, s1, r1, t1, l1 = rm[0]
    print (s0)

    for count in range(len(rm)):
        _, a0, l0, _, _, r1, terminal, l1 = rm[count]
        if terminal or r1:
            print ("index: ", count, a0, l0, r1, terminal, l1)

if __name__ == "__main__":
    coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    args = parser.parse_args()

    test_2(args.env)
