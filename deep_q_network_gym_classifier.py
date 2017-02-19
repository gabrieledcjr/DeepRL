#!/usr/bin/env python
from __future__ import unicode_literals
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import sys
import numpy as np
from data_set import DataSet
from dqn_net_bn_class import DqnNetClass #as Network # with batch normalization
from dqn_net_class import DqnNetClass as Network
import tables
import pandas as pd

try:
    import cPickle as pickle
except ImportError:
    import pickle

# Breakout
GAME = 'breakout' # the name of the game being played for log files
ACTIONS = 4 # number of valid actions

# Pong
# GAME = 'pong' # the name of the game being played for log files
# ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 50000 # timesteps to observe before training
EXPLORE = 1000000 # frames over which to anneal epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 1000000 # number of previous transitions to remember
RESIZED_WIDTH = 84
RESIZED_HEIGHT = 84
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others

PHI_LENGTH = 4 # rms

SAVE_FREQ = 50000
EVAL_FREQ = 5000
EVAL_MAX_STEPS = 10000
C_FREQ = 1000

BATCH_DECAY = 0.9
BATCH_EPSILON = 0.001

OPTIMIZER = 'Adam'
if OPTIMIZER == 'Adam':
    LEARNING_RATE = 0.00025
    DECAY = 0.
    MOMENTUM = 0.
    EPSILON = 0.001
else:
    LEARNING_RATE = 0.00025
    DECAY = 0.95
    MOMENTUM = 0.
    EPSILON = 0.01

PATH = os.getcwd() + '/'
FOLDER = '{}_networks_classifier_{}'.format(GAME, OPTIMIZER.lower())
HUMAN_FOLDER = '{}_human_samples/'.format(GAME)

#try epsilon 1e-3, 1e-8
def train(sess, D):
    net = Network(
        sess, RESIZED_HEIGHT, RESIZED_WIDTH, PHI_LENGTH, ACTIONS, GAME,
        optimizer=OPTIMIZER, learning_rate=LEARNING_RATE, epsilon=EPSILON, decay=DECAY, momentum=MOMENTUM,
        verbose=False, path=PATH, folder=FOLDER)
    print ("Loading data")
    if GAME == 'pong' or GAME == 'breakout':
        # data were pickled using Python 2 which have compatibility issues in Python 3
        data = pickle.load(open('{}{}-dqn-all.pkl'.format(HUMAN_FOLDER, GAME), 'rb'), encoding='latin1')
    else:
        data = pickle.load(open('{}{}-dqn-all.pkl'.format(HUMAN_FOLDER, GAME), 'rb'))
    D.width = data['D.width']
    D.height = data['D.height']
    D.max_steps = data['D.max_steps']
    D.phi_length = data['D.phi_length']
    D.num_actions = data['D.num_actions']
    D.actions = data['D.actions']
    D.rewards = data['D.rewards']
    D.terminal = data['D.terminal']
    D.bottom = data['D.bottom']
    D.top = data['D.top']
    D.size = data['D.size']
    h5file = tables.open_file('{}{}-dqn-images-all.h5'.format(HUMAN_FOLDER, GAME), mode='r')
    D.imgs = h5file.root.images[:]
    h5file.close()
    print ("Data loaded!")

    for i in range(150000):
        s_j_batch, a_batch, _, _, _ = D.random_batch_classifier(32)
        if i%500 == 0:
            result = net.evaluate(s_j_batch, a_batch)
            acc = result[0]
            summary_str = result[1]
            net.add_summary(summary_str, i)
            print ("step %d, training accuracy %g"%(i, acc))
        # perform gradient step
        net.train(s_j_batch, a_batch)

    for i in range(10):
        s_j_batch, a_batch, _, _, _ = D.random_batch_classifier(1000)
        print ("test accuracy %g"%net.evaluate(s_j_batch, a_batch)[0])

    net.save()



NUM_THREADS = 16
def playGame():
    #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, intra_op_parallelism_threads=NUM_THREADS)) as sess:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True, intra_op_parallelism_threads=NUM_THREADS)) as sess:
        with tf.device('/gpu:'+os.environ["CUDA_VISIBLE_DEVICES"]):
            if False: # Deterministic
                rng = np.random.RandomState(123456)
            else:
                rng = np.random.RandomState()
            replay_memory = DataSet(RESIZED_WIDTH, RESIZED_HEIGHT, rng, REPLAY_MEMORY, PHI_LENGTH, ACTIONS)
            train(sess, replay_memory)

def main():
    playGame()

if __name__ == "__main__":
    main()
