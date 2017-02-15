#!/usr/bin/env python
import random
import tables
import numpy as np
import matplotlib.pyplot as plt
import os
from math import sqrt

try:
    import cPickle as pickle
except ImportError:
    import pickle


def egreedy(readout_t, n_actions=-1):
    assert n_actions > 1
    best_indices = [0]
    max_action = readout_t[0]
    for index in range(1, n_actions):
        if readout_t[index] > max_action:
            best_indices = [index]
            max_action = readout_t[index]
        elif readout_t[index] == max_action:
            best_indices.append(index)
    action_index = random.choice(best_indices)
    return action_index

def get_action_index(readout_t, is_random=False, n_actions=-1):
    assert n_actions > 1
    action_index = 0
    if is_random:
        action_index = random.randrange(n_actions)
    else:
        action_index = egreedy(readout_t, n_actions)
    return action_index

def add_human_experiences(D, good_only=False, samp_num='001', name='pong'):
    good_str = ''
    if good_only:
        good_str = '-good'
    data = pickle.load(open(name + '_human_samples/' + samp_num + '/' + name + good_str + '-dqn-all.pkl', 'rb'))
    terminals = data['D.terminal']
    actions = data['D.actions']
    rewards = data['D.rewards']
    h5file = tables.openFile(name + '_human_samples/' + samp_num + '/' + name + good_str + '-dqn-images-all.h5', mode='r')
    imgs = h5file.root.images[:]
    h5file.close()
    print "\tMemory size={}".format(D.size)
    print "\tAdding {} human experiences...".format(data['D.size'])
    for i in range(data['D.size']):
        s = imgs[i]
        a = actions[i]
        r = rewards[i]
        t = terminals[i]
        D.add_sample(s, a, r, t)
    print "\tMemory size={}".format(D.size)


def plot_conv_weights(weights, name, channels_all=True, folder=''):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    :src: https://github.com/grishasergei/conviz/blob/master/conviz.py
    """
    if folder != '':
        folder = folder + "/plots"
    else:
        folder = "./plots"
    # make path to output folder
    plot_dir = os.path.join(folder, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    #fig, axes = plt.subplots(min([grid_r, grid_c]),
    #                         max([grid_r, grid_c]))

    # iterate channels
    for channel in channels:
        # create figure and axes
        fig, axes = plt.subplots(min([grid_r, grid_c]),
                                 max([grid_r, grid_c]))
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        fig.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')
        plt.close(fig)

def plot_conv_output(conv_img, name, folder=''):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    :src: https://github.com/grishasergei/conviz/blob/master/conviz.py
    """
    if folder != '':
        folder = folder + "/plots"
    else:
        folder = "./plots"
    # make path to output folder
    plot_dir = os.path.join(folder, 'conv_output')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    prepare_dir(plot_dir, empty=True)

    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # get number of convolutional filters
    num_filters = conv_img.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate filters
    for l, ax in enumerate(axes.flat):
        # get a single image
        img = conv_img[0, :, :,  l]
        # put it on the grid
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
    # save figure
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')

def get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    :src: https://github.com/grishasergei/conviz/blob/master/utils.py
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]


def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    :src: https://github.com/grishasergei/conviz/blob/master/utils.py
    """
    factors = set()
    for x in xrange(1, int(sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)


def empty_dir(path):
    """
    Delete all files and folders in a directory
    :param path: string, path to directory
    :return: nothing
    :src: https://github.com/grishasergei/conviz/blob/master/utils.py
    """
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print 'Warning: {}'.format(e)


def create_dir(path):
    """
    Creates a directory
    :param path: string
    :return: nothing
    :src: https://github.com/grishasergei/conviz/blob/master/utils.py
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def prepare_dir(path, empty=False):
    """
    Creates a directory if it soes not exist
    :param path: string, path to desired directory
    :param empty: boolean, delete all directory content if it exists
    :return: nothing
    :src: https://github.com/grishasergei/conviz/blob/master/utils.py
    """
    if not os.path.exists(path):
        create_dir(path)

    if empty:
        empty_dir(path)
