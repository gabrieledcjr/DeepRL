# -*- coding: utf-8 -*-
import threading
import numpy as np

import signal
import random
import math
import os
import time

from termcolor import colored
from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

def run_a3c(args):
    """
    python3 a3c.py --gym-env=PongDeterministic-v3 --parallel-size=16 --initial-learn-rate=7e-4 --use-lstm --use-mnih-2015
    """
    if args.use_gpu:
        assert args.cuda_devices != ''
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import tensorflow as tf
    def log_uniform(lo, hi, rate):
        log_lo = math.log(lo)
        log_hi = math.log(hi)
        v = log_lo * (1-rate) + log_hi * rate
        return math.exp(v)

    device = "/cpu:0"
    gpu_options = None
    if args.use_gpu:
        device = "/gpu:"+os.environ["CUDA_VISIBLE_DEVICES"]
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)

    if args.initial_learn_rate == 0:
        initial_learning_rate = log_uniform(
            args.initial-alpha-low,
            args.initial-alpha-high,
            args.initial-alpha-log-rate)
    else:
        initial_learning_rate = args.initial_learn_rate
    print (colored('Initial Learning Rate={}'.format(initial_learning_rate), 'green'))
    time.sleep(2)

    global_t = 0

    stop_requested = False

    game_state = GameState(env_id=args.gym_env)
    action_size = game_state.env.n_actions
    game_state.env.close()
    del game_state.env
    del game_state

    if args.use_lstm:
        GameACLSTMNetwork.use_mnih_2015 = args.use_mnih_2015
        global_network = GameACLSTMNetwork(action_size, -1, device)
    else:
        GameACFFNetwork.use_mnih_2015 = args.use_mnih_2015
        global_network = GameACFFNetwork(action_size, -1, device)


    training_threads = []

    learning_rate_input = tf.placeholder("float")

    grad_applier = RMSPropApplier(
        learning_rate = learning_rate_input,
        decay = args.rmsp_alpha,
        momentum = 0.0,
        epsilon = args.rmsp_epsilon,
        clip_norm = args.grad_norm_clip,
        device = device)

    A3CTrainingThread.log_interval = args.log_interval
    A3CTrainingThread.performance_log_interval = args.performance_log_interval
    A3CTrainingThread.local_t_max = args.local_t_max
    A3CTrainingThread.use_lstm = args.use_lstm
    A3CTrainingThread.action_size = action_size
    A3CTrainingThread.entropy_beta = args.entropy_beta
    A3CTrainingThread.gamma = args.gamma
    A3CTrainingThread.use_mnih_2015 = args.use_mnih_2015
    A3CTrainingThread.env_id = args.gym_env
    for i in range(args.parallel_size):
        training_thread = A3CTrainingThread(
            i, global_network, initial_learning_rate,
            learning_rate_input,
            grad_applier, args.max_time_step,
            device=device)
        training_threads.append(training_thread)

    # prepare session
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
        allow_soft_placement=True)
    sess = tf.Session(config=config)

    init = tf.global_variables_initializer()
    sess.run(init)

    # summary for tensorboard
    score_input = tf.placeholder(tf.int32)
    tf.summary.scalar("score", score_input)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.log_file, sess.graph)

    # init or load checkpoint with saver
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(args.checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print(colored("checkpoint loaded:{}".format(checkpoint.model_checkpoint_path), "green"))
        tokens = checkpoint.model_checkpoint_path.split("-")
        # set global step
        global_t = int(tokens[1])
        print(">>> global step set: ", global_t)
        # set wall time
        wall_t_fname = args.checkpoint_dir + '/' + 'wall_t.' + str(global_t)
        with open(wall_t_fname, 'r') as f:
            wall_t = float(f.read())
    else:
        print(colored("Could not find old checkpoint", "yellow"))
        # set wall time
        wall_t = 0.0


    def train_function(parallel_index):
        nonlocal global_t

        training_thread = training_threads[parallel_index]
        # set start_time
        start_time = time.time() - wall_t
        training_thread.set_start_time(start_time)

        while True:
            if stop_requested:
                break
            if global_t > args.max_time_step:
                break

            diff_global_t = training_thread.process(
                sess, global_t, summary_writer,
                summary_op, score_input)
            global_t += diff_global_t


    def signal_handler(signal, frame):
        nonlocal stop_requested
        print('You pressed Ctrl+C!')
        stop_requested = True

    train_threads = []
    for i in range(args.parallel_size):
        train_threads.append(threading.Thread(target=train_function, args=(i,)))

    signal.signal(signal.SIGINT, signal_handler)

    # set start time
    start_time = time.time() - wall_t

    for t in train_threads:
        t.start()

    print(colored('Press Ctrl+C to stop', 'blue'))
    signal.pause()

    print(colored('Now saving data. Please wait', 'yellow'))

    for t in train_threads:
        t.join()

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    # write wall time
    wall_t = time.time() - start_time
    wall_t_fname = args.checkpoint_dir + '/' + 'wall_t.' + str(global_t)
    with open(wall_t_fname, 'w') as f:
        f.write(str(wall_t))

    saver.save(sess, args.checkpoint_dir + '/' + 'checkpoint', global_step = global_t)
    print (colored('Data saved!', 'green'))
