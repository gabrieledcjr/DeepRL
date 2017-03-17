#!/usr/bin/env python3
import sys
from util import get_compressed_images

try:
    import cPickle as pickle
except ImportError:
    import pickle

class ClassifyDemo(object):
    def __init__(
        self, net, D, name, train_max_steps, batch_size,
        eval_freq, folder):
        """ Initialize Classifying Human Demo Training """
        self.net = net
        self.D = D
        self.name = name
        self.train_max_steps = train_max_steps
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.folder = folder

        self._load_memory()

    def _load_memory(self):
        print ("Loading data")
        if self.name == 'pong' or self.name == 'breakout':
            # data were pickled using Python 2 which have compatibility issues in Python 3
            data = pickle.load(open('{}/{}-dqn-all.pkl'.format(self.folder, self.name), 'rb'), encoding='latin1')
        else:
            data = pickle.load(open('{}/{}-dqn-all.pkl'.format(self.folder, self.name), 'rb'))

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
        self.D.imgs = get_compressed_images('{}/{}-dqn-images-all.h5'.format(self.folder, self.name) + '.gz')
        print ("Data loaded!")

    def run(self):
        max_val = -(sys.maxsize)
        for i in range(self.train_max_steps):
            s_j_batch, a_batch, _, _, _ = self.D.random_batch(self.batch_size)

            if (i % self.eval_freq) == 0:
                result = self.net.evaluate_batch(s_j_batch, a_batch)
                acc = result[0]
                summary_str = result[1]
                self.net.add_summary(summary_str, i)
                print ("step {}, training accuracy {}, max output val {}".format(i, acc, max_val))

            # perform gradient step
            _, _, _, _, output_vals, max_value = self.net.train(s_j_batch, a_batch)

            if max_value > max_val:
                max_val = max_value

        self.net.save(model_max_output_val=max_val)
