#!/usr/bin/env python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import sys
import envs.gym_fun as game
from experiment import Experiment
from dqn_net import DqnNet
from dqn_net_bn import DqnNetBn

# Breakout
GAME = 'breakout' # the name of the game being played for log files
ACTIONS = 4 # number of valid actions

# Pong
# GAME = 'pong' # the name of the game being played for log files
# ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 0 # timesteps to observe before training
EXPLORE = 1000000 # frames over which to anneal epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 1000000 # number of previous transitions to remember
RESIZED_WIDTH = 84
RESIZED_HEIGHT = 84
BATCH = 32 # size of minibatch

PHI_LENGTH = 4 # rms
UPDATE_FREQUENCY = 4

SAVE_FREQ = 125000
EVAL_FREQ = 250000
EVAL_MAX_STEPS = 125000
TRAIN_MAX_STEPS = 7125000

C_FREQ = 10000
SLOW = False
TAU = 1.

OPTIMIZER = 'Graves'
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
FOLDER = '{}_networks_transfer_{}'.format(GAME, OPTIMIZER.lower())
TRANSFER_FOLDER = "{}_networks_classifier_{}/transfer_model".format(GAME, "adam") # always get pretrain model from adam
VERBOSE = False

NUM_THREADS = 16
NORMALIZE = False
def main():
    #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, intra_op_parallelism_threads=NUM_THREADS)) as sess:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True, intra_op_parallelism_threads=NUM_THREADS)) as sess:
        with tf.device('/gpu:'+os.environ["CUDA_VISIBLE_DEVICES"]):
            if NORMALIZE:
                net = DqnNetBn(
                    sess, RESIZED_HEIGHT, RESIZED_WIDTH, PHI_LENGTH, ACTIONS, GAME, gamma=GAMMA, copy_interval=C_FREQ,
                    optimizer=OPTIMIZER, learning_rate=LEARNING_RATE, epsilon=EPSILON, decay=DECAY, momentum=MOMENTUM,
                    verbose=VERBOSE, path=PATH, folder=FOLDER, slow=SLOW, tau=TAU, decay_learning_rate=True, transfer=True, transfer_folder=TRANSFER_FOLDER)
            else:
                net = DqnNet(
                    sess, RESIZED_HEIGHT, RESIZED_WIDTH, PHI_LENGTH, ACTIONS, GAME, gamma=GAMMA, copy_interval=C_FREQ,
                    optimizer=OPTIMIZER, learning_rate=LEARNING_RATE, epsilon=EPSILON, decay=DECAY, momentum=MOMENTUM,
                    verbose=VERBOSE, path=PATH, folder=FOLDER, slow=SLOW, tau=TAU, transfer=True, transfer_folder=TRANSFER_FOLDER)
            experiment = Experiment(
                sess, net, game, RESIZED_HEIGHT, RESIZED_WIDTH, PHI_LENGTH, ACTIONS, BATCH,
                GAME, GAMMA, OBSERVE, EXPLORE, FINAL_EPSILON, INITIAL_EPSILON, REPLAY_MEMORY,
                UPDATE_FREQUENCY, SAVE_FREQ, EVAL_FREQ, EVAL_MAX_STEPS, C_FREQ,
                OPTIMIZER, LEARNING_RATE, EPSILON, DECAY, MOMENTUM, TAU,
                VERBOSE, PATH, FOLDER, SLOW, load_human_memory=True, train_max_steps=TRAIN_MAX_STEPS)
            experiment.run()

if __name__ == "__main__":
    main()
