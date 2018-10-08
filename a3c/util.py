# -*- coding: utf-8 -*-
import random
import tables
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import gzip
import shutil
import sqlite3
import logging

from math import sqrt
from collections import defaultdict

logger = logging.getLogger("a3c")

try:
    import cPickle as pickle
except ImportError:
    import pickle

def solve_weight(numbers):
    # https://stackoverflow.com/questions/38363764/
    # class-weight-for-imbalance-data-in-python-scikit-learns-logistic-regression
    # n_samples / (n_classes * np.bincount(y))
    # (total # of sample) / ((# of classes) * (# of sample in class i))
    sum_number = sum(numbers)
    len_number = len(numbers)
    solved = [sum_number / (len_number * (n+1e-20)) for n in numbers]
    return solved

def load_memory(name=None, demo_memory_folder=None, imgs_normalized=False, rewards_propagated=False, exclude_outlier_reward=False):
    from replay_memory import ReplayMemory
    assert os.path.isfile(demo_memory_folder + '/demo.db')
    conn = sqlite3.connect(
        demo_memory_folder + '/demo.db',
        detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
    db = conn.cursor()
    replay_buffers = []
    logger.info("Loading data from memory")
    total_memory = 0
    actions_ctr = defaultdict(int)
    max_reward = 0
    max_reward_norm = 0.
    total_rewards = defaultdict(float)
    for demo in db.execute("SELECT * FROM demo_samples"):
        logger.info(demo)
        if name is None:
            name = demo[2]
        ep = demo[1]
        total_memory += demo[4]
        folder = demo_memory_folder + '/{n:03d}'.format(n=(ep))
        logger.info(folder + '/' + name + '-dqn.pkl')
        replay_memory = ReplayMemory()
        replay_memory.load(name=name, folder=folder)
        if imgs_normalized:
            replay_memory.normalize_images()

        temp_max_reward = np.linalg.norm(replay_memory.rewards, np.inf)
        if temp_max_reward > max_reward:
            max_reward = temp_max_reward

        if exclude_outlier_reward:
            rewards = replay_memory.rewards[np.nonzero(replay_memory.rewards)]
            rewards = rewards[np.abs(rewards - np.mean(rewards)) < 2 * np.std(rewards)]
            if np.shape(rewards)[0] > 0:
                temp_max_reward_norm = np.linalg.norm(rewards, np.inf)
            else:
                temp_max_reward_norm = max_reward
            if temp_max_reward_norm > max_reward_norm:
                max_reward_norm = temp_max_reward_norm

        for step in range(len(replay_memory)):
            _, a, r, _, _, _, _ = replay_memory[step]
            actions_ctr[a] += 1
            total_rewards[len(replay_buffers)] += r
        replay_buffers.append(replay_memory)

    if rewards_propagated:
        for i in range(len(replay_buffers)):
            #replay_buffers[i].propagate_rewards(normalize=True, exclude_outlier=exclude_outlier_reward, max_reward=max_reward_norm)
            replay_buffers[i].propagate_rewards(clip=True)

    logger.info("Replay Buffers size: {}".format(len(replay_buffers)))
    logger.info("Total memory: {}".format(total_memory))
    logger.info("Data loaded!")
    conn.close()
    return replay_buffers, actions_ctr, max_reward, total_rewards

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
    for x in range(1, int(sqrt(n)) + 1):
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
            logger.warn('Warning: {}'.format(e))

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

#This code allows gifs to be saved of the training episode for use in the Control Center.
def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
    """
    src: https://github.com/awjuliani/DeepRL-Agents/blob/master/helper.py
    """
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)

    def make_mask(t):
        try:
            x = salIMGS[int(len(salIMGS)/duration*t)]
        except:
            x = salIMGS[-1]
        return x

    clip = mpy.VideoClip(make_frame, duration=duration)
    if salience == True:
        mask = mpy.VideoClip(make_mask, ismask=True,duration= duration)
        clipB = clip.set_mask(mask)
        clipB = clip.set_opacity(0)
        mask = mask.set_opacity(0.1)
        mask.write_gif(fname, fps = len(images) / duration,verbose=False)
        #clipB.write_gif(fname, fps = len(images) / duration,verbose=False)
    else:
        clip.write_gif(fname, fps = len(images) / duration,verbose=False)

def process_frame42(frame):
    frame = frame[34:34+160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    #frame = np.reshape(frame, [np.prod(frame.shape)])
    return frame

def process_frame84(frame):
    frame = frame[34:34+160, :160]
    frame = cv2.resize(frame, (84, 84))
    frame = frame.mean(2)
    frame = frame.astype(np.uint8)
    #frame *= (1.0 / 255.0)
    #frame = np.reshape(frame, [84, 84, 1])
    #frame = np.reshape(frame, [np.prod(frame.shape)])
    return frame

def process_frame(frame, h, w):
    frame = frame[34:34+160, :160]
    frame = cv2.resize(frame, (h, w))
    frame = frame.mean(2)
    frame = frame.astype(np.uint8)
    #frame *= (1.0 / 255.0)
    #frame = np.reshape(frame, [84, 84, 1])
    #frame = np.reshape(frame, [np.prod(frame.shape)])
    return frame

def compress_h5file(file_h5, gz_compress_level=1):
    with open(file_h5, 'rb') as f_in, gzip.open(file_h5 + '.gz', 'wb', gz_compress_level) as f_out:
        shutil.copyfileobj(f_in, f_out)
    return file_h5 + '.gz'

def uncompress_h5file(file_h5):
    import uuid
    temp_file = str(uuid.uuid4()) + '.h5'
    with gzip.open(file_h5, 'rb') as f_in:
        f_out = open(temp_file, 'wb')
        shutil.copyfileobj(f_in, f_out)
        f_out.close()
        h5file = tables.open_file(temp_file, mode='r')
    return h5file, temp_file

def save_compressed_images(file_h5, imgs):
    h5file = tables.open_file(file_h5, mode='w', title='Images Array')
    root = h5file.root
    h5file.create_array(root, "images", imgs)
    h5file.close()
    gz_file = compress_h5file(file_h5)
    remove_h5file(file_h5)
    return gz_file

def get_compressed_images(h5file_gz):
    h5file, temp_file = uncompress_h5file(h5file_gz)
    imgs = h5file.root.images[:]
    h5file.close()
    remove_h5file(temp_file)
    return imgs

def remove_h5file(file_h5):
    os.remove(file_h5)

def get_activations(sess, layer, s_t, s, keep_prob):
  units = sess.run(layer, feed_dict={s: [s_t], keep_prob:1.0})
  plot_nnfilter(units)

def plot_nnfilter(units):
  import matplotlib.pyplot as plt
  import math
  filters = units.shape[3]
  plt.figure(1, figsize=(20,20))
  n_columns = 6
  n_rows = math.ceil(filters / n_columns) + 1
  for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")

def montage(W):
    """Draws all filters (n_input * n_output filters) as a
    montage image separated by 1 pixel borders.
    Parameters
    ----------
    W : numpy.ndarray
        Input array to create montage of.
    Returns
    -------
    m : numpy.ndarray
        Montage image.
    src: https://github.com/pkmital/tensorflow_tutorials/blob/master/python/libs/utils.py
    """
    W = np.reshape(W, [W.shape[0], W.shape[1], 1, W.shape[2] * W.shape[3]])
    n_plots = int(np.ceil(np.sqrt(W.shape[-1])))
    m = np.ones(
        (W.shape[0] * n_plots + n_plots + 1,
         W.shape[1] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < W.shape[-1]:
                m[1 + i + i * W.shape[0]:1 + i + (i + 1) * W.shape[0],
                  1 + j + j * W.shape[1]:1 + j + (j + 1) * W.shape[1]] = (
                    np.squeeze(W[:, :, :, this_filter]))
    return m
