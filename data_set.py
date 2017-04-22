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
    """
    This replay memory assumes it's a single episode of memory in sequence
    """
    def __init__(self,
        width=1, height=1, rng=np.random.RandomState(),
        max_steps=10, phi_length=4, num_actions=1):
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

        self.size = 0

    def resize(self):
        print ("Resizing replay memory...")
        print ("Current specs:")
        print ("\tsize:{}".format(self.size))
        print ("\tmax_steps:{}".format(self.max_steps))
        print ("\timgs shape:", np.shape(self.imgs))
        tmp_imgs = np.delete(self.imgs, range(self.size,self.max_steps), axis=0)
        del self.imgs
        self.imgs = tmp_imgs
        self.max_steps = self.size
        print ("Resizing completed!")
        print ("Updated specs:")
        print ("\tsize:{}".format(self.size))
        print ("\tmax_steps:{}".format(self.max_steps))
        print ("\timgs shape:", np.shape(self.imgs))

    def add_sample(self, img, action, reward, terminal):
        """Add a time step record.

        Arguments:
            img -- observed image
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
        if self.size == self.max_steps:
            print ("Memory is full. Data not added!")
            return
        self.imgs[self.size] = img
        self.actions[self.size] = action
        self.rewards[self.size] = reward
        self.terminal[self.size] = terminal
        self.size += 1

    def __len__(self):
        """Return an approximate count of stored state transitions."""
        # TODO: Properly account for indices which can't be used, as in
        # random_batch's check.
        return max(0, self.size - self.phi_length)

    def __getitem__(self, key):
        indices = np.arange(key, key + self.phi_length)
        end_index = key + self.phi_length - 1

        state = np.zeros((84, 84, self.phi_length), dtype=np.uint8)
        temp = self.imgs.take(indices, axis=0)
        for i in range(self.phi_length):
            state[:, :, i] = temp[i]

        action = self.actions.take(end_index, axis=0)
        reward = self.rewards.take(end_index)
        terminal = self.terminal.take(end_index)

        return state, action, reward, terminal

    def __str__(self):
        specs = "Replay memory:\n"
        specs += "  size:{}\n".format(self.size)
        specs += "  max_steps:{}\n".format(self.max_steps)
        specs += "  imgs shape:{}\n".format(np.shape(self.imgs))
        return specs

    def get_item(self, key):
        return self.__getitem__(key)

    def random_batch(self, batch_size):
        """Return corresponding states, actions, rewards, terminal status, and
        next_states for batch_size randomly chosen state transitions.

        """
        # Allocate the response.
        states = np.zeros((batch_size, 84, 84, self.phi_length), dtype=np.uint8)
        actions = np.zeros((batch_size, self.num_actions), dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        terminals = np.zeros(batch_size, dtype=np.int)

        # Randomly choose a time step from the replay memory
        # within requested batch_size
        end_range = self.size - (self.phi_length-1 + batch_size)
        assert end_range > 0 # crash if not enough memory
        self.random_index = self.rng.randint(0, end_range)
        index = self.random_index
        #end_index = (index + self.phi_length-1) + (batch_size - 1)

        for count in range(batch_size):
            s, a, r, t = self[index]
            # Add the state transition to the response.
            states[count] = np.copy(s)
            actions[count][a] = 1. # convert to one-hot vector
            rewards[count] = r
            terminals[count] = t
            index += 1

        return states, actions, rewards, terminals

def test_1(env_id):
    from util import get_compressed_images
    try:
        import cPickle as pickle
    except ImportError:
        import pickle

    folder = env_id.replace('-', '_') + "_test_demo_samples"
    D = DataSet()
    data = pickle.load(open(folder + '/001/' + env_id + '-dqn.pkl', 'rb'))
    D.width = data['D.width']
    D.height = data['D.height']
    D.max_steps = data['D.max_steps']
    D.phi_length = data['D.phi_length']
    D.num_actions = data['D.num_actions']
    D.actions = data['D.actions']
    D.rewards = data['D.rewards']
    D.terminal = data['D.terminal']
    D.size = data['D.size']
    D.imgs = get_compressed_images(folder + '/001/' + env_id + '-dqn-images.h5' + '.gz')
    print (D)

    for i in range(1000):
        D.random_batch(20)
        print (D.random_index)

    import cv2
    count = 0
    state, _, _, _ = D[count]
    print ("shape:", np.shape(state))
    while count < len(D):
        state, _, _, _ = D[count]
        cv2.imshow(env_id, state)
        # cv2.imshow("one", state[:,:,0])
        # cv2.imshow("two", state[:,:,1])
        # cv2.imshow("three", state[:,:,2])
        # cv2.imshow("four", state[:,:,3])
        diff1 = cv2.absdiff(state[:,:,3], state[:,:,2])
        diff2 = cv2.absdiff(state[:,:,3], state[:,:,1])
        diff3 = cv2.absdiff(state[:,:,3], state[:,:,0])
        diff = cv2.addWeighted(diff1, 0.8, diff2, 0.2, 0.0)
        diff = cv2.addWeighted(diff, 0.8, diff3, 0.2, 0.0)
        cv2.imshow("difference", diff)
        cv2.waitKey(20)
        count += 1
    print ("total transitions:", len(D))
    print ("size:", D.size)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    args = parser.parse_args()

    test_1(args.env)
