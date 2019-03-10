#!/usr/bin/env python3
"""Replay memory module.

Copyright (c) 2014, Nathan Sprague
All rights reserved.

Original code: https://goo.gl/dp2qRV
This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.
"""
import coloredlogs
import logging
import numpy as np
import random

from collections import defaultdict
from common.util import get_compressed_images
from common.util import save_compressed_images
from common.util import transform_h
from common.util import transform_h_inv

try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger("replay_memory")


class ReplayMemory(object):
    """Replay Memory Class.

    This replay memory assumes it's a single episode of memory in sequence
    """

    def __init__(self, width=1, height=1, rng=np.random.RandomState(),
                 max_steps=10, phi_length=4, num_actions=1, wrap_memory=False,
                 full_state_size=1013):
        """Construct a replay memory.

        Keyword arguments:
        width, height -- image size
        max_steps -- the number of time steps to store
        phi_length -- number of images to concatenate into a state
        rng -- initialized numpy random number generator
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
        self.full_state = np.zeros((self.max_steps, full_state_size),
                                   dtype=np.uint8)

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

    def resize(self):
        if self.max_steps == self.size:
            return
        logger.info("Resizing replay memory...")
        logger.debug("Current specs: size={} max_steps={}".format(
            self.size, self.max_steps))
        logger.debug("    images shape: {}".format(np.shape(self.imgs)))
        logger.debug("    actions shape: {}".format(np.shape(self.actions)))
        logger.debug("    rewards shape: {}".format(np.shape(self.rewards)))
        logger.debug("    terminal shape: {}".format(np.shape(self.terminal)))
        logger.debug("    lives shape: {}".format(np.shape(self.lives)))
        logger.debug("    full_state shape: {}".format(
            np.shape(self.full_state)))

        del_range = range(self.size, self.max_steps)
        tmp_imgs = np.delete(self.imgs, del_range, axis=0)
        tmp_actions = np.delete(self.actions, del_range, axis=0)
        tmp_rewards = np.delete(self.rewards, del_range, axis=0)
        tmp_terminal = np.delete(self.terminal, del_range, axis=0)
        tmp_lives = np.delete(self.lives, del_range, axis=0)
        tmp_fullstate = np.delete(self.full_state, del_range, axis=0)

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
        logger.debug("Updated specs: size={}"
                     " max_steps={}".format(self.size, self.max_steps))
        logger.debug("    images shape: {}".format(np.shape(self.imgs)))
        logger.debug("    actions shape: {}".format(np.shape(self.actions)))
        logger.debug("    rewards shape: {}".format(np.shape(self.rewards)))
        logger.debug("    terminal shape: {}".format(np.shape(self.terminal)))
        logger.debug("    lives shape: {}".format(np.shape(self.lives)))
        logger.debug("    full_state shape: {}".format(
            np.shape(self.full_state)))

    def add(self, img, action, reward, terminal, lives, fullstate=None):
        """Add a time step record.

        Storing in replay memory should follow the following format:
        s0 = [img0, img1, img2, img3]
        s1 = [img1, img2, img3, img4]
        img0 | img1 | img2 | img3 | img4  <= image memory
                           |   s0 |   s1  <= state memory
                           |      |   a0  <= action memory
                           |      |   r1  <= reward memory
                           |      |   t1  <= terminal memory
                           |   l0 |   l1  <= lives memory
                           |  fs0 |  fs1  <= full state memory

        Keyword arguments:
        img -- observed image
        action -- action chosen by the agent
        reward -- reward received after taking the action
        terminal -- boolean indicating whether the episode ended after this
            time step
        lives -- number of lives remaining
        fullstate -- ALE environment's full state
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
        return max(0, self.size - self.phi_length - 1)

    def get_item(self, key, check_terminal=True):
        """Retrieve data using key from replay memory.

        Follows the following format:
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

        mode = 'wrap' if self.wrap_memory else 'raise'
        if check_terminal:
            if self.wrap_memory:
                if np.any(self.terminal.take(indices, mode='wrap')):
                    return None, None, None, None, None, None, None, None
            else:
                if end_index >= self.size \
                   or np.any(self.terminal.take(indices)):
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

    def __getitem__(self, key):
        """Retrieve data using key from replay memory."""
        return self.get_item(key, check_terminal=False)

    def __str__(self):
        """Output string info about the class."""
        specs = "Replay memory:\n"
        specs += "  size:{}\n".format(self.size)
        specs += "  max_steps:{}\n".format(self.max_steps)
        specs += "  imgs shape:{}\n".format(np.shape(self.imgs))
        specs += "  wrap memory:{}\n".format(self.wrap_memory)
        return specs

    def sample_sequential(self, batch_size):
        """Return a sample in sequence."""
        assert not self.wrap_memory

        # Allocate the response.
        st_shape = (batch_size, self.height, self.width, self.phi_length)
        dtype = np.float32 if self.imgs_normalized else np.uint8
        states = np.zeros(st_shape, dtype=dtype)
        actions = np.zeros((batch_size, self.num_actions), dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        terminals = np.zeros(batch_size, dtype=np.int)
        # lives = np.zeros(batch_size, dtype=np.int)

        # Randomly choose a time step from the replay memory
        # within requested batch_size
        # randint low (inclusive) to high (exclusive)
        high = (self.size + 1) - (self.phi_length + batch_size)
        assert high > 0  # crash if not enough memory

        # ensure no terminal besides the last index
        while True:
            self.random_index = self.rng.randint(0, high)
            indices = np.arange(
                self.random_index,
                self.random_index + batch_size + self.phi_length)

            if not np.any(self.terminal.take(indices[:-1])):
                index = self.random_index
                break

        for count in range(batch_size):
            s0, a0, l0, fs0, s1, r1, t1, l1 = self.get_item(index)
            assert s0 is not None
            # Add the state transition to the response.
            states[count] = np.copy(s0)
            actions[count][a0] = 1.  # convert to one-hot vector
            rewards[count] = r1
            terminals[count] = t1
            index += 1

        return states, actions, rewards, terminals

    def create_index_array_per_action(self):
        """Cluster indices for each action into an array."""
        assert not self.wrap_memory
        self.array_per_action = defaultdict(list)
        for index in range(len(self)):
            s0, a0, l0, fs0, s1, r1, t1, l1 = self.get_item(index)
            if s0 is None or s1 is None:
                continue
            self.array_per_action[a0].append(index)

    def random_batch_actions(self, batch_size, action_distribution, type=None):
        """Sample proportional or oversample.

        Types:
        oversample -- sampling equally from all actions
        proportional -- sample based on proportion to the action distribution
        None -- randomly sample from any action
        """
        assert type == 'oversample' or type == 'proportional'

        a = range(len(action_distribution))
        if type == 'oversample':
            num_nonzeros = np.count_nonzero(action_distribution)
            mask = np.clip(action_distribution, 0, 1)
            proportions = np.ones(len(action_distribution), dtype=np.float32)
            proportions /= num_nonzeros
            proportions *= mask
        else:
            # Proportional
            proportions = action_distribution / np.sum(action_distribution)

        sample = np.random.choice(a, size=batch_size, p=proportions)
        return sample

    def sample(self, batch_size, onevsall=False, n_class=None, reward_type=''):
        """Return a random sample in a wrap memory."""
        assert self.wrap_memory

        # Allocate the response.
        st_shape = (batch_size, self.height, self.width, self.phi_length)
        states = np.zeros(st_shape, dtype=np.uint8)
        next_states = np.zeros(st_shape, dtype=np.uint8)
        rewards = np.zeros(batch_size, dtype=np.float32)
        terminals = np.zeros(batch_size, dtype=np.int)
        # lives = np.zeros(batch_size, dtype=np.int)

        if onevsall:
            actions = np.zeros((batch_size, 2), dtype=np.float32)
        else:
            actions = np.zeros((batch_size, self.num_actions),
                               dtype=np.float32)

        high = self.bottom + self.size - self.phi_length
        assert high > 0  # crash if not enough memory

        count = 0
        while count < batch_size:
            index = self.rng.randint(self.bottom, high)
            s0, a0, l0, fs0, s1, r1, t1, l1 = self.get_item(index)
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
                actions[count][a0] = 1  # convert to one-hot vector
            if reward_type == 'CLIP':
                r1 = np.sign(r1)
            elif reward_type == 'LOG':
                r1 = np.sign(r1) * np.log(1. + np.abs(r1))
            rewards[count] = r1
            terminals[count] = t1
            count += 1

        return states, actions, rewards, terminals, next_states

    def save(self, name=None, folder=None, resize=False):
        """Save data to file."""
        assert name is not None
        assert folder is not None

        if resize:
            # Resize replay memory to exact memory size
            self.resize()

        data = {
            'width': self.width,
            'height': self.height,
            'max_steps': self.max_steps,
            'phi_length': self.phi_length,
            'num_actions': self.num_actions,
            'actions': self.actions,
            'rewards': self.rewards,
            'terminal': self.terminal,
            'lives': self.lives,
            'full_state_size': self.full_state_size,
            'full_state': self.full_state,
            'size': self.size,
            'wrap_memory': self.wrap_memory,
            'top': self.top,
            'bottom': self.bottom,
            'imgs_normalized': self.imgs_normalized,
            }

        images = self.imgs
        pkl_file = folder / '{}.pkl'.format(name)
        h5_file = folder / '{}-images.h5'.format(name)
        pickle.dump(data, pkl_file.open('wb'), pickle.HIGHEST_PROTOCOL)
        logger.info('Compressing and saving replay memory...')
        save_compressed_images(h5_file, images)
        logger.info('Compressed and saved replay memory')

    def load(self, name=None, folder=None):
        """Load data from file."""
        assert name is not None
        assert folder is not None

        logger.info('Load memory from {}...'.format(folder))
        pkl_file = folder / '{}.pkl'.format(name)
        h5_file = folder / '{}-images.h5.gz'.format(name)
        data = pickle.load(pkl_file.open('rb'))
        self.width = data['width']
        self.height = data['height']
        self.max_steps = data['max_steps']
        self.phi_length = data['phi_length']
        self.num_actions = data['num_actions']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.terminal = data['terminal']
        self.lives = data.get(
            'lives', np.zeros(self.max_steps, dtype=np.int32))
        self.full_state_size = data.get(
            'full_state_size', self.full_state_size)
        self.full_state = data.get(
            'full_state', np.zeros((self.max_steps, self.full_state_size),
                                   dtype=np.uint8))
        self.size = data['size']
        self.wrap_memory = data['wrap_memory']
        self.top = data['top']
        self.bottom = data['bottom']
        self.imgs_normalized = data['imgs_normalized']
        self.imgs = get_compressed_images(h5_file)


class ReplayMemoryReturns(ReplayMemory):
    """Replay Memory with Returns Class.

    This replay memory assumes it's a single episode of memory in sequence
    """

    def __init__(self, width=1, height=1, rng=np.random.RandomState(),
                 max_steps=10, phi_length=4, num_actions=1, wrap_memory=False,
                 full_state_size=1013, gamma=0.99, clip=False):
        """Construct a replay memory."""
        ReplayMemory.__init__(self, width, height, rng, max_steps, phi_length,
                              num_actions, wrap_memory, full_state_size)
        self.returns = None
        self.gamma = gamma
        self.clip = clip

    @staticmethod
    def compute_returns(rewards, terminals, gamma, clip=False, c=1.89):
        """Compute expected return."""
        length = np.shape(rewards)[0]
        returns = np.empty_like(rewards, dtype=np.float32)

        if clip:
            rewards = np.clip(rewards, -1., 1.)
        else:
            # when reward is 1, t(r=1) = 0.412 which is less than half of
            # reward which slows down the training with Atari games with
            # raw rewards at range (-1, 1). To address this down scaled reward,
            # we add the constant c=sign(r) * 1.89 to ensure that
            # t(r=1 + sign(r) * 1.89) ~ 1
            rewards = np.sign(rewards) * c + rewards

        assert terminals[-1]  # assert that last state is a terminal state
        for i in reversed(range(length)):
            if terminals[i]:
                returns[i] = rewards[i] if clip else transform_h(rewards[i])
            else:
                if clip:
                    returns[i] = rewards[i] + gamma * returns[i+1]
                else:
                    # apply transformed expected return
                    exp_r_t = gamma * transform_h_inv(returns[i+1])
                    returns[i] = transform_h(rewards[i] + exp_r_t)
        return returns

    def compute_expected_returns(self):
        """Compute expected return."""
        assert not self.wrap_memory
        logger.info("Propagating rewards...")
        logger.info("reward size: {}".format(np.shape(self.rewards)[0]))
        logger.info("gamma: {}".format(self.gamma))
        logger.info("clip: {}".format(self.clip))

        self.returns = self.__class__.compute_returns(
            self.rewards, self.terminal, self.gamma, clip=self.clip)

    def _sample_by_indices(self, batch_size, onevsall=False, n_class=None):
        # Allocate the response.
        st_shape = (batch_size, self.height, self.width, self.phi_length)
        states = np.zeros(st_shape, dtype=np.uint8)
        rewards = np.zeros(batch_size, dtype=np.float32)
        terminals = np.zeros(batch_size, dtype=np.int)
        returns = np.zeros(batch_size, dtype=np.float32)

        if onevsall:
            actions = np.zeros((batch_size, 2), dtype=np.float32)
        else:
            actions = np.zeros((batch_size, self.num_actions),
                               dtype=np.float3)

        # Randomly choose a time step from the replay memory
        # within requested batch_size
        # randint low (inclusive) to high (exclusive)
        high = self.size - self.phi_length
        assert high > 0  # crash if not enough memory

        count = 0
        while count < batch_size:
            index = self.rng.randint(0, high)

            s0, a0, l0, fs0, s1, r1, t1, l1 = self.get_item(index)
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
                actions[count][a0] = 1  # convert to one-hot vector
            rewards[count] = r1
            terminals[count] = t1
            returns[count] = self.returns.take(index + self.phi_length)
            count += 1

        return states, actions, rewards, terminals, returns

    def _sample_by_actions(self, batch_size, action_distribution, type=None,
                           onevsall=False, n_class=None):
        # Allocate the response.
        st_shape = (batch_size, self.height, self.width, self.phi_length)
        states = np.zeros(st_shape, dtype=np.uint8)
        rewards = np.zeros(batch_size, dtype=np.float32)
        terminals = np.zeros(batch_size, dtype=np.int)
        returns = np.zeros(batch_size, dtype=np.float32)

        if onevsall:
            actions = np.zeros((batch_size, 2), dtype=np.float32)
        else:
            actions = np.zeros((batch_size, self.num_actions),
                               dtype=np.float32)

        random_actions = self.random_batch_actions(batch_size,
                                                   action_distribution,
                                                   type=type)

        count = 0
        while count < batch_size:
            action = random_actions[count]
            index = random.choice(self.array_per_action[action])
            s0, a0, l0, fs0, s1, r1, t1, l1 = self.get_item(index)
            assert action == a0

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
                actions[count][a0] = 1  # convert to one-hot vector
            rewards[count] = r1
            terminals[count] = t1
            returns[count] = self.returns.take(index + self.phi_length)
            count += 1

        return states, actions, rewards, terminals, returns

    def sample_nowrap(self, batch_size, action_distribution, type=None,
                      onevsall=False, n_class=None):
        """Return a random sample based on the type in a no wrap memory."""
        assert not self.wrap_memory

        if self.array_per_action is None:
            self.create_index_array_per_action()

        if self.returns is None:
            self.compute_expected_returns()

        if type is None:
            return self._sample_by_indices(batch_size, onevsall, n_class)

        return self._sample_by_actions(batch_size, action_distribution, type,
                                       onevsall, n_class)


def test_1(env_id):
    """Test 1."""
    folder = "demo_samples/{}".format(env_id.replace('-', '_'))
    rm = ReplayMemory()
    rm.load(name=env_id, folder=(folder + '/001'))
    print(rm)

    for i in range(1000):
        states, actions, rewards, terminals = rm.sample2(20)

    import cv2
    count = 0
    state, _, _, _, _, _, _, _ = rm[count]
    print("shape:", np.shape(state))
    while count < len(rm):
        state, _, _, _, _, _, _, _ = rm[count]
        cv2.imshow(env_id, state)
        cv2.imshow("one", state[:, :, 0])
        cv2.imshow("two", state[:, :, 1])
        cv2.imshow("three", state[:, :, 2])
        cv2.imshow("four", state[:, :, 3])
        # diff1 = cv2.absdiff(state[:,:,3], state[:,:,2])
        # diff2 = cv2.absdiff(state[:,:,3], state[:,:,1])
        # diff3 = cv2.absdiff(state[:,:,3], state[:,:,0])
        # diff = cv2.addWeighted(diff1, 0.8, diff2, 0.2, 0.0)
        # diff = cv2.addWeighted(diff, 0.8, diff3, 0.2, 0.0)
        # cv2.imshow("difference", diff)
        cv2.waitKey(20)
        count += 1
    print("total transitions:", len(rm))
    print("size:", rm.size)


def test_2(env_id):
    """Test 2."""
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
    print(s0)

    for count in range(len(rm)):
        _, a0, l0, _, _, r1, terminal, l1 = rm[count]
        if terminal or r1:
            print("index: ", count, a0, l0, r1, terminal, l1)


if __name__ == "__main__":
    fmt = "%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s"
    coloredlogs.install(level='DEBUG', fmt=fmt)
    logger.setLevel(logging.DEBUG)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    args = parser.parse_args()

    test_2(args.env)
