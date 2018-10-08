"""
Copyright (c) 2014, Nathan Sprague
All rights reserved.

Original code: https://goo.gl/dp2qRV

This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.
"""

import numpy as np
import time
import logging

from util import save_compressed_images

try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger("a3c")

class DataSet(object):
    """
    This replay memory assumes it's a single episode of memory in sequence
    """
    def __init__(self,
        width=1, height=1, rng=np.random.RandomState(),
        max_steps=10, phi_length=4, num_actions=1, wrap_memory=False):
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
        self.lives = np.zeros(self.max_steps, dtype=np.int32)
        self.loss_life = np.zeros(self.max_steps, dtype=np.uint8)
        self.gain_life = np.zeros(self.max_steps, dtype=np.uint8)

        self.size = 0
        self.imgs_normalized = False

        self.wrap_memory = wrap_memory
        self.bottom = 0
        self.top = 0

    def normalize_images(self):
        if not self.imgs_normalized:
            logger.info("Normalizing images...")
            temp = self.imgs
            self.imgs = temp * (1.0/255.0)
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
        logger.info("Resizing replay memory...")
        logger.debug("Current specs: size={} max_steps={}".format(self.size, self.max_steps))
        logger.debug("    images shape: {}".format(np.shape(self.imgs)))
        logger.debug("    actions shape: {}".format(np.shape(self.actions)))
        logger.debug("    rewards shape: {}".format(np.shape(self.rewards)))
        logger.debug("    terminal shape: {}".format(np.shape(self.terminal)))
        logger.debug("    lives shape: {}".format(np.shape(self.lives)))
        logger.debug("    loss_life shape: {}".format(np.shape(self.loss_life)))
        logger.debug("    gain_life shape: {}".format(np.shape(self.gain_life)))
        tmp_imgs = np.delete(self.imgs, range(self.size,self.max_steps), axis=0)
        tmp_actions = np.delete(self.actions, range(self.size, self.max_steps), axis=0)
        tmp_rewards = np.delete(self.rewards, range(self.size, self.max_steps), axis=0)
        tmp_terminal = np.delete(self.terminal, range(self.size, self.max_steps), axis=0)
        tmp_lives = np.delete(self.lives, range(self.size, self.max_steps), axis=0)
        tmp_losslife = np.delete(self.loss_life, range(self.size, self.max_steps), axis=0)
        tmp_gainlife = np.delete(self.gain_life, range(self.size, self.max_steps), axis=0)
        del self.imgs, self.actions, self.rewards, self.terminal, self.lives, self.loss_life
        self.imgs = tmp_imgs
        self.actions = tmp_actions
        self.rewards = tmp_rewards
        self.terminal = tmp_terminal
        self.lives = tmp_lives
        self.loss_life = tmp_losslife
        self.gain_life = tmp_gainlife
        self.max_steps = self.size
        logger.info("Resizing completed!")
        logger.debug("Updated specs: size={} max_steps={}".format(self.size, self.max_steps))
        logger.debug("    images shape: {}".format(np.shape(self.imgs)))
        logger.debug("    actions shape: {}".format(np.shape(self.actions)))
        logger.debug("    rewards shape: {}".format(np.shape(self.rewards)))
        logger.debug("    terminal shape: {}".format(np.shape(self.terminal)))
        logger.debug("    lives shape: {}".format(np.shape(self.lives)))
        logger.debug("    loss_life shape: {}".format(np.shape(self.loss_life)))
        logger.debug("    gain_life shape: {}".format(np.shape(self.gain_life)))

    # def fix_size(self):
    #     if self.max_steps == self.size:
    #         if np.shape(self.actions)[0] > self.size:
    #             max_size = np.shape(self.actions)[0]
    #             tmp_actions = np.delete(self.actions, range(self.size, max_size), axis=0)
    #             del self.actions
    #             self.actions = tmp_actions
    #         if np.shape(self.rewards)[0] > self.size:
    #             max_size = np.shape(self.rewards)[0]
    #             tmp_rewards = np.delete(self.rewards, range(self.size, max_size), axis=0)
    #             del self.rewards
    #             self.rewards = tmp_rewards
    #         if np.shape(self.terminal)[0] > self.size:
    #             max_size = np.shape(self.terminal)[0]
    #             tmp_terminal = np.delete(self.terminal, range(self.size, max_size), axis=0)
    #             del self.terminal
    #             self.terminal = tmp_terminal

    def add_sample(self, img, action, reward, terminal, lives, losslife=False, gainlife=False):
        """Add a time step record.

        Arguments:
            img -- observed image
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
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
        self.loss_life[idx] = losslife
        self.gain_life[idx] = gainlife

        if self.wrap_memory == self.max_steps and self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1

        if self.wrap_memory:
            self.top = (self.top + 1) % self.max_steps

    def __len__(self):
        """Return an approximate count of stored state transitions."""
        # TODO: Properly account for indices which can't be used, as in
        # random_batch's check.
        return max(0, self.size - self.phi_length)

    def __getitem__(self, key):
        indices = np.arange(key, key + self.phi_length)
        end_index = key + self.phi_length - 1
        state = np.zeros(
            (self.height, self.width, self.phi_length),
            dtype=np.float32 if self.imgs_normalized else np.uint8)

        if self.wrap_memory:
            if np.any(self.terminal.take(indices[0:self.phi_length-1], mode='wrap')):
                return None, None, None, None, None
            temp = self.imgs.take(indices, axis=0, mode='wrap')
            for i in range(self.phi_length):
                states[:, :, i] = termp[i]
        else:
            temp = self.imgs.take(indices, axis=0)
            for i in range(self.phi_length):
                state[:, :, i] = temp[i]

        action = self.actions.take(end_index, axis=0)
        reward = self.rewards.take(end_index+1)
        terminal = self.terminal.take(end_index+1)
        lives = self.lives.take(end_index+1)
        losslife = self.loss_life.take(end_index+1)
        gainlife = self.gain_life.take(end_index+1)

        return state, action, reward, terminal, lives, losslife, gainlife

    def __str__(self):
        specs = "Replay memory:\n"
        specs += "  size:{}\n".format(self.size)
        specs += "  max_steps:{}\n".format(self.max_steps)
        specs += "  imgs shape:{}\n".format(np.shape(self.imgs))
        return specs

    def get_item(self, key):
        return self.__getitem__(key)

    def random_batch_sequential(self, batch_size):
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
        # losslifes = np.zeros(batch_size, dtype=np.int)
        # gainlifes = np.zeros(batch_size, dtype=np.int)

        # Randomly choose a time step from the replay memory
        # within requested batch_size
        end_range = self.size - (self.phi_length-1 + batch_size)
        assert end_range > 0 # crash if not enough memory
        self.random_index = self.rng.randint(0, end_range)
        index = self.random_index
        #end_index = (index + self.phi_length-1) + (batch_size - 1)

        for count in range(batch_size):
            s, a, r, t, l, ll, gl = self[index]
            # Add the state transition to the response.
            states[count] = np.copy(s)
            actions[count][a] = 1. # convert to one-hot vector
            rewards[count] = r
            terminals[count] = t
            # lives[count] = l
            # losslifes[count] = ll
            # gainlifes[count] = gl
            index += 1

        return states, actions, rewards, terminals #, lives, losslifes, gainlifes

    def random_batch(self, batch_size, normalize=False, k_bad_states=0, onevsall=False, n_class=None):
        """Return corresponding states, actions, rewards, terminal status, and
        next_states for batch_size randomly chosen state transitions.

        """
        # Allocate the response.
        states = np.zeros(
            (batch_size, self.height, self.width, self.phi_length),
            dtype=np.float32 if (normalize or self.imgs_normalized) else np.uint8)
        if onevsall:
            actions = np.zeros((batch_size, 2), dtype=np.float32)
        else:
            actions = np.zeros((batch_size, self.num_actions), dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        terminals = np.zeros(batch_size, dtype=np.int)
        # lives = np.zeros(batch_size, dtype=np.int)
        # losslifes = np.zeros(batch_size, dtype=np.int)
        # gainlifes = np.zeros(batch_size, dtype=np.int)

        # Randomly choose a time step from the replay memory
        # within requested batch_size
        end_range = self.size - (self.phi_length + 1)
        assert end_range > 0 # crash if not enough memory

        count = 0
        while count < batch_size:
            index = self.rng.randint(0, end_range)
            if k_bad_states:
                # do not train k steps to a bad state (negative reward or loss life)
                st_idx = index + (self.phi_length-1) + 1
                en_idx = index + (self.phi_length-1) + 1 + k_bad_states
                if (np.any(self.rewards[st_idx:en_idx] < 0) or \
                    np.any(self.loss_life[st_idx:en_idx] == 1)):
                    continue

            s, a, r, t, l, ll, gl = self[index]
            # Add the state transition to the response.
            states[count] = np.copy(s)
            if normalize and not self.imgs_normalized:
                states[count] *= (1.0/255.0)
            if onevsall:
                if a == n_class:
                    actions[count][0] = 1
                else:
                    actions[count][1] = 1
            else:
                actions[count][a] = 1 # convert to one-hot vector
            rewards[count] = r
            terminals[count] = t
            # lives[count] = l
            # losslifes[count] = ll
            # gainlifes[count] = gl
            count += 1

        return states, actions, rewards, terminals #, lives, losslifes, gainlifes

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
                'loss_life':self.loss_life,
                'gain_life':self.gain_life,
                'size':self.size,
                'wrap_memory':self.wrap_memory,
                'top':self.top,
                'bottom':self.bottom,
                'imgs_normalized':self.imgs_normalized}
        images = self.imgs
        pkl_file = '{}-dqn.pkl'.format(name)
        h5_file = '{}-dqn-images.h5'.format(name)
        pickle.dump(data, open(folder + pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)
        logger.info('Compressing and saving replay memory...')
        save_compressed_images(folder + h5_file, images)
        logger.info('Compressed and saved replay memory')

    def load(self, name=None, folder=None):
        assert name is not None
        assert folder is not None

        from util import get_compressed_images
        try:
            import cPickle as pickle
        except ImportError:
            import pickle

        data = pickle.load(open(folder + '/' + name + '-dqn.pkl', 'rb'))
        self.width = data['width']
        self.height = data['height']
        self.max_steps = data['max_steps']
        self.phi_length = data['phi_length']
        self.num_actions = data['num_actions']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.terminal = data['terminal']
        self.lives = data['lives'] if 'lives' in data else np.zeros(D.max_steps, dtype=np.int32)
        self.loss_life = data['loss_life'] if 'loss_life' in data else np.zeros(D.max_steps, dtype=np.uint8)
        self.gain_life = data['gain_life'] if 'gain_life' in data else np.zeros(D.max_steps, dtype=np.uint8)
        self.size = data['size']
        self.wrap_memory = data['wrap_memory']
        self.top = data['top']
        self.bottom = data['bottom']
        self.imgs_normalized = data['imgs_normalized']
        self.imgs = get_compressed_images(folder + '/' + name + '-dqn-images.h5' + '.gz')


def test_1(env_id):
    from util import get_compressed_images
    try:
        import cPickle as pickle
    except ImportError:
        import pickle

    folder = env_id.replace('-', '_') + "_test_demo_samples"
    D = DataSet()
    D.load(name=env_id, folder=(folder + '/001'))
    #data = pickle.load(open(folder + '/001/' + env_id + '-dqn.pkl', 'rb'))
    print (D)

    for i in range(1000):
        states, actions, rewards, terminals, lives, losslife, gainlife = D.random_batch(20)

    import cv2
    count = 0
    state, _, _, _, _, _, _ = D[count]
    print ("shape:", np.shape(state))
    while count < len(D):
        state, _, _, _, _ = D[count]
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

def test_2(env_id):
    from util import get_compressed_images
    try:
        import cPickle as pickle
    except ImportError:
        import pickle

    folder = "demo_samples/{}".format(env_id.replace('-', '_'))
    D = DataSet()
    D.load(name=env_id, folder=(folder + '/001'))
    print (D)

    state, a, r, t, l, ll, gl = D[0]
    print (state)
    print (a, r, t, l, ll, gl)

    D.normalize_images()
    state, a, r, t, l, ll, gl = D[2]
    print (state)
    print (a, r, t, l, ll, gl)

    for count in range(100):
        _, a, r, t, l, ll, gl = D[count]
        print (a,r,t,l,ll,gl)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    args = parser.parse_args()

    test_2(args.env)
