"""
Copyright (c) 2014, Nathan Sprague
All rights reserved.

Original code: https://goo.gl/dp2qRV

This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.
"""

import numpy as np
import time

class DataSet(object):
    """A replay memory consisting of circular buffers for observed images,
    actions, and rewards.
    """
    def __init__(self, width, height, rng, max_steps=1000, phi_length=4, num_actions=1):
        """Construct a DataSet.

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

        # Allocate the circular buffers and indices.
        self.imgs = np.zeros((self.max_steps, height, width), dtype=np.uint8)
        self.actions = np.zeros(self.max_steps, dtype=np.uint8)
        self.rewards = np.zeros(self.max_steps, dtype=np.float32)
        self.terminal = np.zeros(self.max_steps, dtype=np.uint8)

        self.bottom = 0
        self.top = 0
        self.size = 0

        self.validation_set_markers = None
        self.validation_indices = None
        self.validation_set_initialize = False

    def resize(self):
        print ("Resizing replay memory...")
        print ("Current specs:")
        print ("\tsize:{}".format(self.size))
        print ("\tmax_steps:{}".format(self.max_steps))
        print ("\ttop:{}".format(self.top))
        print ("\tbottom:{}".format(self.bottom))
        print ("\timgs shape:", np.shape(self.imgs))
        tmp_imgs = np.delete(self.imgs, range(self.size,self.max_steps), axis=0)
        del self.imgs
        self.imgs = tmp_imgs
        self.top = 0
        self.bottom = 1
        self.max_steps = self.size
        print ("Resizing completed!")
        print ("Updated specs:")
        print ("\tsize:{}".format(self.size))
        print ("\tmax_steps:{}".format(self.max_steps))
        print ("\ttop:{}".format(self.top))
        print ("\tbottom:{}".format(self.bottom))
        print ("\timgs shape:", np.shape(self.imgs))

    def create_validation_set(self, percent=0.2):
        print ("Creating validation set...")
        self.validation_set_markers = np.zeros(self.max_steps, dtype=np.uint8)
        self.validation_indices = []
        self.validation_set_size = int(self.size * percent)
        total_validation_set = self.validation_set_size
        print ("Validation Set: {} of {}".format(total_validation_set, self.size))
        while total_validation_set > 0:
            index = np.random.randint(4, self.size-4)

            # cannot be a terminal state since no action will be takened
            # and it cannot be in the validation set already
            indices = np.arange(index, index + self.phi_length)
            if self.validation_set_markers[index] == 1 or np.any(self.terminal.take(indices, mode='wrap')):
                continue

            self.validation_set_markers[index] = 1
            self.validation_indices.append(index)
            total_validation_set -= 1
        print ("Validation set created!")

    def add_sample(self, img, action, reward, terminal):
        """Add a time step record.

        Arguments:
            img -- observed image
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
        self.imgs[self.top] = img
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminal[self.top] = terminal

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def __len__(self):
        """Return an approximate count of stored state transitions."""
        # TODO: Properly account for indices which can't be used, as in
        # random_batch's check.
        return max(0, self.size - self.phi_length)

    def last_phi(self):
        """Return the most recent phi (sequence of image frames)."""
        indexes = np.arange(self.top - self.phi_length, self.top)
        return self.imgs.take(indexes, axis=0, mode='wrap')

    def phi(self, img):
        """Return a phi (sequence of image frames), using the last phi_length -
        1, plus img.

        """
        indexes = np.arange(self.top - self.phi_length + 1, self.top)

        phi = np.empty((self.phi_length, self.height, self.width), dtype='float')
        phi[0:self.phi_length - 1] = self.imgs.take(indexes,
                                                    axis=0,
                                                    mode='wrap')
        phi[-1] = img
        return phi

    def init_validation_set(self):
        validation_set_size = len(self.validation_indices)
        # Allocate the response.
        self.validation_set_states = np.zeros((validation_set_size, 84, 84, self.phi_length), dtype=np.uint8)
        self.validation_set_actions = np.zeros((validation_set_size, self.num_actions), dtype=np.float32)

        for count, index in enumerate(self.validation_indices):
            indices = np.arange(index, index + self.phi_length)
            end_index = index + self.phi_length - 1

            # Add the state to validation set
            temp = self.imgs.take(indices, axis=0, mode='wrap')
            for i in range(self.phi_length):
                self.validation_set_states[count, :, :, i] = temp[i]

            a_idx = self.actions.take(end_index, axis=0, mode='wrap')
            self.validation_set_actions[count][a_idx] = 1.

    def get_validation_set(self):
        if not self.validation_set_initialize:
            self.init_validation_set()
            self.validation_set_initialize = True
        return self.validation_set_states, self.validation_set_actions

    def random_batch(self, batch_size, exclude_validation=False):
        """Return corresponding states, actions, rewards, terminal status, and
        next_states for batch_size randomly chosen state transitions.

        """
        # Allocate the response.
        states = np.zeros((batch_size, 84, 84, self.phi_length), dtype=np.uint8)
        actions = np.zeros((batch_size, self.num_actions), dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        terminal = np.zeros(batch_size, dtype=np.int)
        next_states = np.zeros((batch_size, 84, 84, self.phi_length), dtype=np.uint8)

        count = 0
        while count < batch_size:
            # Randomly choose a time step from the replay memory.
            index = self.rng.randint(self.bottom,
                                     self.bottom + self.size - self.phi_length)

            indices = np.arange(index, index + self.phi_length + 1)
            end_index = index + self.phi_length - 1

            # Check that the initial state corresponds entirely to a
            # single episode, meaning none but the last frame may be
            # terminal. If the last frame of the initial state is
            # terminal, then the last frame of the transitioned state
            # will actually be the first frame of a new episode, which
            # the Q learner recognizes and handles correctly during
            # training by zeroing the discounted future reward estimate.
            if np.any(self.terminal.take(indices[0:self.phi_length-1], mode='wrap')):
                continue

            # excluding validation set from random batch
            if exclude_validation and (index in self.validation_indices):
                continue

            # Add the state transition to the response.
            temp = self.imgs.take(indices, axis=0, mode='wrap')
            for i in range(self.phi_length):
                states[count, :, :, i] = temp[i]
            next_states[count] = np.copy(states[count])
            next_states[count] = np.roll(next_states[count], -1, axis=2)
            next_states[count, :, :, -1] = temp[self.phi_length]

            a_idx = self.actions.take(end_index, axis=0, mode='wrap')
            actions[count][a_idx] = 1.
            rewards[count] = self.rewards.take(end_index, mode='wrap')
            terminal[count] = self.terminal.take(end_index, mode='wrap')

            count += 1

        return states, actions, rewards, next_states, terminal
