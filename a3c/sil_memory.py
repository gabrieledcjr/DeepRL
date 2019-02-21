#!/usr/bin/env python3
import cv2
import logging
import numpy as np
import random

from common.util import transform_h
from common.util import transform_h_inv

logger = logging.getLogger("sil_memory")


class SILReplayMemory(object):

    def __init__(self, num_actions, max_len=None, gamma=0.99, clip=False,
                 height=84, width=84, phi_length=4):
        self.num_actions = num_actions
        self.states = []
        self.actions = []
        self.rewards = []
        self.terminal = []
        self.returns = []
        self.maxlen = max_len
        self.gamma = gamma
        self.clip = clip
        self.height = height
        self.width = width
        self.phi_length = phi_length

    def add_item(self, s, a, rew, t):
        """Use only for episode memory."""
        assert len(self.returns) == 0
        if np.shape(s) != self.shape():
            s = cv2.resize(s, (self.height, self.width),
                           interpolation=cv2.INTER_AREA)
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(rew)
        self.terminal.append(t)

    def reset(self):
        """Reset memory."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.terminal.clear()
        self.returns.clear()

    def shape(self):
        """Return shape of state."""
        return (self.height, self.width, self.phi_length)

    def extend(self, x):
        """Use only in SIL memory."""
        self.states.extend(x.states)
        self.actions.extend(x.actions)
        self.rewards.extend(x.rewards)

        assert x.terminal[-1]  # assert that last state is a terminal state
        x_returns = self.__class__.compute_returns(
            x.rewards, x.terminal, self.gamma, self.clip)

        self.returns.extend(x_returns)
        self.terminal.extend(x.terminal)

        if len(self) > self.maxlen:
            st_slice = len(self) - self.maxlen
            self.states = self.states[st_slice:]
            self.actions = self.actions[st_slice:]
            self.rewards = self.rewards[st_slice:]
            self.returns = self.returns[st_slice:]
            self.terminal = self.terminal[st_slice:]
            assert len(self) == self.maxlen

        x.reset()
        assert len(x) == 0
        assert len(self) == len(self.returns) <= self.maxlen

    @staticmethod
    def compute_returns(rewards, terminal, gamma, clip=False, c=1.89):
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
            rewards = np.clip(rewards, -1., 1.) * c + rewards

        # # if all zeros
        # if not np.any(rewards):
        #     logger.info("episode all zeros")
        #     rewards[-1] = -1. * c + -1.

        for i in reversed(range(length)):
            if terminal[i]:
                returns[i] = rewards[i] if clip else transform_h(rewards[i])
            else:
                if clip:
                    returns[i] = rewards[i] + gamma * returns[i+1]
                else:
                    # apply transformed expected return
                    exp_r_t = gamma * transform_h_inv(returns[i+1])
                    returns[i] = transform_h(rewards[i] + exp_r_t)
        return returns

    def __len__(self):
        """Return length of memory using states."""
        return len(self.states)

    def sample(self, batch_size):
        """Return a random batch sample from the memory."""
        assert len(self.states) >= batch_size

        if len(self.returns) == 0:
            returns = self.__class__.compute_returns(
                self.rewards, self.terminal, self.gamma, self.clip)
            self.returns.extend(returns)

        shape = (batch_size, self.height, self.width, self.phi_length)
        states = np.zeros(shape, dtype=np.uint8)
        actions = np.zeros((batch_size, self.num_actions), dtype=np.float32)
        returns = np.zeros(batch_size, dtype=np.float32)

        random_indices = random.sample(range(0, len(self.states)), batch_size)
        for i, rand_i in enumerate(random_indices):
            states[i] = np.copy(self.states[rand_i])
            actions[i][self.actions[rand_i]] = 1  # one-hot vector
            returns[i] = self.returns[rand_i]

        return states, actions, returns

    def __del__(self):
        """Clean up."""
        del self.states
        del self.actions
        del self.rewards
        del self.terminal
        del self.returns
