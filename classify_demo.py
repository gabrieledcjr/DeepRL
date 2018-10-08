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
        eval_freq, demo_memory_folder='', folder=''):
        """ Initialize Classifying Human Demo Training """
        self.net = net
        self.D = D
        self.name = name
        self.train_max_steps = train_max_steps
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.demo_memory_folder = demo_memory_folder
        self.folder = folder

        self._load_memory()

    def _load_memory(self):
        print ("Loading data")
        if self.name == 'pong' or self.name == 'breakout':
            # data were pickled using Python 2 which have compatibility issues in Python 3
            data = pickle.load(open('{}/{}-dqn-all.pkl'.format(self.demo_memory_folder, self.name), 'rb'), encoding='latin1')
        else:
            data = pickle.load(open('{}/{}-dqn-all.pkl'.format(self.demo_memory_folder, self.name), 'rb'))

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
        self.D.validation_set_markers = data['D.validation_set_markers']
        self.D.validation_indices = data['D.validation_indices']
        self.D.imgs = get_compressed_images('{}/{}-dqn-images-all.h5'.format(self.demo_memory_folder, self.name) + '.gz')
        print ("Data loaded!")

    def run(self):
        data = {
            'training_step': [],
            'training_accuracy': [],
            'training_entropy': [],
            'testing_step': [],
            'testing_accuracy': [],
            'testing_entropy': [],
            'max_accuracy': 0.,
            'max_accuracy_step': 0,
        }
        max_val = -(sys.maxsize),
        no_change_ctr = 0

        s_j_batch_validation, a_batch_validation = self.D.get_validation_set()
        for i in range(self.train_max_steps):
            s_j_batch, a_batch, _, _, _ = self.D.random_batch(self.batch_size, exclude_validation=True)

            if (i % self.eval_freq) == 0:
                entropy, acc, _, _ = self.net.evaluate_batch(s_j_batch_validation, a_batch_validation)
                data['testing_step'].append(i)
                data['testing_accuracy'].append(acc)
                data['testing_entropy'].append(entropy)

                if acc > data['max_accuracy']:
                    data['max_accuracy'] = acc
                    data['max_accuracy_step'] = i
                    # early stopping (save best model)
                    self.net.save(model_max_output_val=max_val, step=data['max_accuracy_step'])
                    no_change_ctr = 0
                else:
                    no_change_ctr += 1

                self.net.add_accuracy(acc, entropy, i, stage='Validation')
                print ("step {}, max accuracy {}, testing accuracy {}, no change ctr {}, max output val {}".format(i, data['max_accuracy'], acc, no_change_ctr, max_val))

            # UNCOMMENT BEFORE PUSHING TO GITHUB
            # if no_change_ctr == 100:
            #     break

            # perform gradient step
            _, entropy, acc, output_vals, max_value = self.net.train(s_j_batch, a_batch)
            data['training_step'].append(i)
            data['training_accuracy'].append(acc)
            data['training_entropy'].append(entropy)
            if (i % self.eval_freq) == 0:
                print ("\tstep {}, training accuracy {}".format(i, acc))

            self.net.add_accuracy(acc, entropy, i, stage='Training')

            if max_value > max_val:
                max_val = max_value

        self.net.save(model_max_output_val=max_val, relative='/final/')
        pickle.dump(data, open(self.folder + '/data', 'wb'), pickle.HIGHEST_PROTOCOL)
        print ("final max output val {}".format(max_val))

    def save_max_value(self, max_val=-(sys.maxsize)):
        batch = self.D.size * 10 // 100
        for i in range(100):
            s_j_batch, a_batch, _, _, _ = self.D.random_batch(batch)
            _, _, output_vals, max_value = self.net.evaluate_batch(s_j_batch, a_batch)
            if i%10 == 0:
                print ("step {}, max output val {}".format(i, max_val))

            if max_value > max_val:
                print ("Max value from {} to {}".format(max_val, max_value))
                max_val = max_value

        print ("max output val {}".format(max_val))
        self.net.save_max_value(max_val)
