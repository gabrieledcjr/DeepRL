#!/usr/bin/env python3
import random
import tables
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import cv2
import gzip
import shutil
import sqlite3
import logging

from math import sqrt
from collections import defaultdict

logger = logging.getLogger("util")


def percent_decrease(v1, v2):
    """Compute percent difference.

    old_value (v1) - new_value (v2)
    -------------------------------  * 100%
          | old_value (v1) |
    """
    return (v2 - v1) / (abs(v1) + 1e-10) * 100

def transform_h(z, eps=10**-2):
    return (np.sign(z) * (np.sqrt(np.abs(z) + 1.) - 1.)) + (eps * z)

def transform_h_inv(z, eps=10**-2):
    return np.sign(z) * (np.square((np.sqrt(1 + 4 * eps * (np.abs(z) + 1 + eps)) - 1) / (2 * eps)) - 1)

def transform_h_log(z, eps=.6):
    return (np.sign(z) * np.log(1. + np.abs(z)) * eps)

def transform_h_inv_log(z, eps=.6):
    return np.sign(z) * (np.exp(np.abs(z) / eps) - 1)

def grad_cam(activations, gradients):
    # global average pooling
    weights = np.mean(gradients, axis=(0, 1))  # 64
    cam = np.zeros(activations.shape[0:2], dtype=np.float32)  # 7, 7

    # Modified Grad-CAM
    # Summing and rectifying weighted activations across depth
    for i, w in enumerate(weights):
        # only care about positive w (ReLU)
        cam += np.maximum(w, 0.) * activations[:, :, i]

    return cam

def visualize_cam(cam):
    # create heatmap image for cam
    if np.max(cam) > 0:
        cam = cam / np.max(cam) # scale to 0 to 1.0
    cam = cv2.resize(cam, (84, 84))

    cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)

    return cam_heatmap

def generate_image_for_cam_video(state_img, cam_img, global_t, img_index, action):
    cam_img = np.uint8(cam_img)

    # create one state
    mean_state = np.mean(state_img[:,:,0:3], axis=-1)
    state = np.maximum(state_img[:,:,3], mean_state)
    state = np.uint8(state)

    state_rgb = cv2.cvtColor(state, cv2.COLOR_GRAY2RGB)

    # add information text to output video
    info = np.zeros((84, 110, 3), dtype=np.uint8)
    cv2.putText(info, "Step#{}".format(global_t),
        (3, 15), cv2.FONT_HERSHEY_DUPLEX, .4, (255, 255, 255), 1)
    cv2.putText(info, "Frame#{}".format(img_index),
        (3, 30), cv2.FONT_HERSHEY_DUPLEX, .4, (255, 255, 255), 1)
    cv2.putText(info, "{}".format(action),
        (3, 45), cv2.FONT_HERSHEY_DUPLEX, .4, (255, 255, 255), 1)

    # overlay cam-state
    # alpha = 0.5
    # output = cv2.addWeighted(cam_img, alpha, state_rgb, 1 - alpha, 0)
    # overlay_output = cv2.hconcat((output, info))

    # side-by-side cam-state
    hcat_cam_state =  cv2.hconcat((cam_img, state_rgb))
    vcat_title_camstate = cv2.hconcat((hcat_cam_state, info))

    return vcat_title_camstate

def solve_weight(numbers):
    # https://stackoverflow.com/questions/38363764/
    # class-weight-for-imbalance-data-in-python-scikit-learns-logistic-regression
    # n_samples / (n_classes * np.bincount(y))
    # (total # of sample) / ((# of classes) * (# of sample in class i))
    sum_number = sum(numbers)
    len_number = len(numbers)
    #solved = [sum_number / (len_number * (n+1e-20)) for n in numbers]
    solved = [sum_number / (len_number * (n+1)) for n in numbers]

    return solved

def load_memory(name=None, demo_memory_folder=None, demo_ids=None):
    assert demo_ids is not None
    assert demo_memory_folder is not None

    from common.replay_memory import ReplayMemory
    assert os.path.isfile(str(demo_memory_folder / 'demo.db'))

    logger.info("Loading data from memory")
    logger.info("memory_folder: {}".format(demo_memory_folder))
    logger.info("demo_ids: {}".format(demo_ids))

    conn = sqlite3.connect(
        str(demo_memory_folder / 'demo.db'),
        detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
    db = conn.cursor()
    replay_buffers = {}
    total_memory = 0
    action_distribution = defaultdict(int)
    total_rewards = defaultdict(float)
    total_steps = 0

    for demo in db.execute("SELECT * FROM demo_samples WHERE id IN ({})".format(demo_ids)):
        # logger.info(demo)
        if name is None:
            name = demo[2]
        demo_id = demo[0]
        datetime_collected = demo[1]
        total_rewards[demo_id] = demo[5]
        total_memory += demo[6]
        hostname = demo[13]

        folder = demo_memory_folder / 'data' / hostname / str(datetime_collected)
        replay_memory = ReplayMemory()
        replay_memory.load(name=name, folder=folder)
        total_steps += replay_memory.max_steps

        actions_count = np.unique(replay_memory.actions, return_counts=True)
        for index, action in enumerate(actions_count[0]):
            action_distribution[action] += actions_count[1][index]

        replay_buffers[demo_id] = replay_memory

    logger.info("replay_buffers size: {}".format(len(replay_buffers)))
    logger.info("total_rewards: {}".format(dict.__repr__(total_rewards)))
    logger.info("total_memory: {}".format(total_memory))
    logger.info("total_steps: {}".format(total_steps))
    logger.info("action_distribution: {}".format(dict.__repr__(action_distribution)))
    logger.info("Data loaded!")
    conn.close()
    return replay_buffers, action_distribution, total_rewards, total_steps

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

def graves_rmsprop_optimizer(loss, learning_rate, rmsprop_decay, rmsprop_constant, gradient_clip):
    """
    src:https://raw.githubusercontent.com/cgel/DRL/master/agents/commonOps.py
    """
    import tensorflow as tf
    with tf.name_scope('rmsprop'):
        optimizer = None
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        grads_and_vars = optimizer.compute_gradients(loss)

        grads = []
        params = []
        for p in grads_and_vars:
            if p[0] == None:
                continue
            grads.append(p[0])
            params.append(p[1])
        #grads = [gv[0] for gv in grads_and_vars]
        #params = [gv[1] for gv in grads_and_vars]
        if gradient_clip > 0:
            grads = tf.clip_by_global_norm(grads, gradient_clip)[0]

        square_grads = [tf.square(grad) for grad in grads]

        avg_grads = [tf.Variable(tf.zeros(var.get_shape()))
                     for var in params]
        avg_square_grads = [tf.Variable(
            tf.zeros(var.get_shape())) for var in params]

        update_avg_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + tf.scalar_mul((1 - rmsprop_decay), grad_pair[1]))
                            for grad_pair in zip(avg_grads, grads)]
        update_avg_square_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * tf.square(grad_pair[1])))
                                   for grad_pair in zip(avg_square_grads, grads)]
        avg_grad_updates = update_avg_grads + update_avg_square_grads

        rms = [tf.sqrt(avg_grad_pair[1] - tf.square(avg_grad_pair[0]) + rmsprop_constant)
               for avg_grad_pair in zip(avg_grads, avg_square_grads)]

        rms_updates = [grad_rms_pair[0] / grad_rms_pair[1]
                       for grad_rms_pair in zip(grads, rms)]
        train = optimizer.apply_gradients(zip(rms_updates, params))

        return tf.group(train, tf.group(*avg_grad_updates)), grads_and_vars

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
    if path is not str:
        path = str(path)

    if not os.path.exists(path):
        create_dir(path)

    if empty:
        empty_dir(path)

#This code allows gifs to be saved of the training episode for use in the Control Center.
def make_movie(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
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
        mask = mpy.VideoClip(make_mask, ismask=True, duration=duration)
        clipB = clip.set_mask(mask)
        clipB = clip.set_opacity(0)
        mask = mask.set_opacity(0.1)
        mask.write_videofile(fname + ".mp4", fps=24)
        #mask.write_gif(fname + ".gif", fps=(len(images) / duration), verbose=False)
    else:
        clip.write_videofile(fname + ".mp4", fps=24)
        #clip.write_gif(fname + ".gif", fps=(len(images) / duration), verbose=False)

def process_frame42(frame):
    frame = frame[34:34+160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame /= 255.0
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
    with file_h5.open('rb') as f_in, gzip.open(str(file_h5.with_suffix('.h5.gz')), 'wb', gz_compress_level) as f_out:
        shutil.copyfileobj(f_in, f_out)
    return file_h5.with_suffix('.h5.gz')

def uncompress_h5file(file_h5):
    import uuid
    temp_file = pathlib.Path(str(uuid.uuid4()) + '.h5')
    with gzip.open(str(file_h5), 'rb') as f_in:
        f_out = temp_file.open('wb')
        shutil.copyfileobj(f_in, f_out)
        f_out.close()
        h5file = tables.open_file(str(temp_file), mode='r')
    return h5file, temp_file

def save_compressed_images(file_h5, imgs):
    h5file = tables.open_file(str(file_h5), mode='w', title='Images Array')
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
    file_h5.unlink()

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
