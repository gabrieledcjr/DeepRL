# -*- coding: utf-8 -*-
import threading
import numpy as np

import signal
import random
import math
import os
import time

from termcolor import colored
from util import load_memory
from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

try:
    import cPickle as pickle
except ImportError:
    import pickle

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

    if not os.path.exists('results'):
        os.mkdir('results')

    if args.folder is not None:
        folder = 'results/{}_{}'.format(args.gym_env.replace('-', '_'), args.folder)
    else:
        folder = 'results/{}'.format(args.gym_env.replace('-', '_'))

    end_str = ''
    if args.use_transfer:
        end_str += '_transfer'
        if args.not_transfer_fc2:
            end_str += '_nofc2'
    if args.train_with_demo_epoch > 0:
        end_str += '_pretrain_ina3c'
    if args.set_last_threads_as_demo:
        end_str += '_demothreads'
    if args.use_mnih_2015:
        end_str += '_use_mnih'
    if args.use_lstm:
        end_str += '_use_lstm'
    folder += end_str

    if args.append_experiment_num is not None:
        folder += '_' + args.append_experiment_num

    demo_memory = None
    if args.load_memory:
        if args.demo_memory_folder is not None:
            demo_memory_folder = 'demo_samples/{}'.format(args.demo_memory_folder)
        else:
            demo_memory_folder = 'demo_samples/{}'.format(args.gym_env.replace('-', '_'))
        demo_memory, actions_ctr = load_memory(args.gym_env, demo_memory_folder, imgs_normalized=True)

    device = "/cpu:0"
    gpu_options = None
    if args.use_gpu:
        device = "/gpu:"+os.environ["CUDA_VISIBLE_DEVICES"]
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)

    if args.initial_learn_rate == 0:
        initial_learning_rate = log_uniform(
            args.initial_alpha_low,
            args.initial_alpha_high,
            args.initial_alpha_log_rate)
    else:
        initial_learning_rate = args.initial_learn_rate
    print (colored('Initial Learning Rate={}'.format(initial_learning_rate), 'green'))
    time.sleep(2)

    global_t = 0
    pretrain_global_t = 0
    pretrain_epoch = 0
    testing_rewards = {}
    training_rewards = {}

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

    if args.use_transfer:
        if args.transfer_folder is not None:
            transfer_folder = args.transfer_folder
        else:
            transfer_folder = '{}_classifier'.format(args.gym_env.replace('-', '_'))
            end_str = ''
            if args.use_mnih_2015:
                end_str += '_use_mnih'
            # if args.use_lstm:
            #     end_str += '_use_lstm'
            transfer_folder += end_str
            transfer_folder += '/transfer_model'

        transfer_var_list = [
            global_network.W_conv1, global_network.b_conv1,
            global_network.W_conv2, global_network.b_conv2,
            global_network.W_fc1, global_network.b_fc1
        ]
        if args.use_mnih_2015:
            transfer_var_list += [
                global_network.W_conv3, global_network.b_conv3
            ]
        if not args.not_transfer_fc2 and not args.use_lstm:
            transfer_var_list += [
                global_network.W_fc2, global_network.b_fc2
            ]
        global_network.load_transfer_model(
            sess, folder=transfer_folder,
            transfer_fc2=(False if args.not_transfer_fc2 or args.use_lstm else True),
            var_list=transfer_var_list
        )

    def initialize_uninitialized(sess):
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

        #print [str(i.name) for i in not_initialized_vars] # only for testing
        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))

    if args.use_transfer:
        initialize_uninitialized(sess)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    # summary for tensorboard
    score_input = tf.placeholder(tf.float32)
    tf.summary.scalar("score", score_input)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('log/{}/'.format(args.gym_env.replace('-', '_')) + folder[8:], sess.graph)

    # init or load checkpoint with saver
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(folder)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print(colored("checkpoint loaded:{}".format(checkpoint.model_checkpoint_path), "green"))
        tokens = checkpoint.model_checkpoint_path.split("-")
        # set global step
        global_t = int(tokens[1])
        print(">>> global step set: ", global_t)
        # set wall time
        wall_t_fname = folder + '/' + 'wall_t.' + str(global_t)
        with open(wall_t_fname, 'r') as f:
            wall_t = float(f.read())
        with open(folder + '/pretrain_global_t', 'r') as f:
            pretrain_global_t = int(f.read())
        testing_rewards = pickle.load(open(folder + '/' + args.gym_env.replace('-', '_') + '-rewards.pkl', 'rb'))
        training_rewards = pickle.load(open(folder + '/' + args.gym_env.replace('-', '_') + '-train-rewards.pkl', 'rb'))
    else:
        print(colored("Could not find old checkpoint", "yellow"))
        # set wall time
        wall_t = 0.0

    lock = threading.Lock()
    test_lock = False
    if global_t == 0:
        test_lock = True

    last_temp_global_t = global_t
    ispretrain_markers = [False] * args.parallel_size
    def train_function(parallel_index):
        nonlocal global_t, pretrain_global_t, pretrain_epoch, \
            testing_rewards, training_rewards, test_lock, lock, \
            last_temp_global_t, ispretrain_markers
        training_thread = training_threads[parallel_index]

        # 1/8 of the threads is used to simulate demo memory
        if args.load_memory and (args.parallel_size - (args.parallel_size/8)) <= parallel_index:
            training_thread.is_demo_thread = args.set_last_threads_as_demo

        if global_t == 0 and args.train_with_demo_epoch > 0:
            ispretrain_markers[parallel_index] = True
            training_thread.pretrain_init(demo_memory)

            # Pretraining with demo memory
            print ("t_idx={} pretrain starting".format(parallel_index))
            while ispretrain_markers[parallel_index]:
                if stop_requested:
                    break
                if pretrain_epoch > args.train_with_demo_epoch:
                    # At end of pretraining, reset state
                    training_thread.episode_reward = 0
                    training_thread.local_t = 0
                    if args.use_lstm:
                        training_thread.local_network.reset_state()
                    ispretrain_markers[parallel_index] = False
                    print ("t_idx={} pretrain ended".format(parallel_index))
                    break

                diff_pretrain_global_t = training_thread.demo_process(sess, pretrain_global_t)
                for _ in range(diff_pretrain_global_t):
                    pretrain_global_t += 1

                pretrain_epoch += 1
                if pretrain_epoch % 1000 == 0:
                    print ("pretrain_epoch={} pretrain_global_t={}".format(pretrain_epoch, pretrain_global_t))

            # Waits for all threads to finish pretraining
            while not stop_requested and any(ispretrain_markers):
                time.sleep(0.01)

        # Evaluate model before training
        if not stop_requested and global_t == 0:
            with lock:
                if parallel_index == 0:
                    test_reward = training_threads[0].testing(
                        sess, args.eval_max_steps, global_t,
                        summary_writer)
                    testing_rewards[global_t] = test_reward
                    test_lock = False
            # all threads wait until evaluation finishes
            while not stop_requested and test_lock:
                time.sleep(0.01)

        if training_thread.is_demo_thread:
            training_thread.pretrain_init(demo_memory)

        # set start_time
        start_time = time.time() - wall_t
        training_thread.set_start_time(start_time)
        while True:
            if stop_requested:
                break
            if global_t > (args.max_time_step * args.max_time_step_fraction):
                break

            if training_thread.is_demo_thread:
                if global_t < args.max_steps_threads_as_demo:
                    diff_global_t = training_thread.demo_process(
                        sess, global_t,
                        pretrain_global_t=pretrain_global_t)
                else:
                    diff_global_t = 0
                    training_thread.episode_reward = 0
                    if args.use_lstm:
                        training_thread.local_network.reset_state()
                    training_thread.is_demo_thread = False
            else:
                diff_global_t = training_thread.process(
                    sess, global_t, summary_writer,
                    summary_op, score_input, training_rewards,
                    pretrain_global_t)

            for _ in range(diff_global_t):
                global_t += 1
                if global_t % args.eval_freq == 0:
                    temp_global_t = global_t
                    lock.acquire()
                    try:
                        # catch multiple threads getting in at the same time
                        if last_temp_global_t == temp_global_t:
                            print(colored("Threading race problem averted!", "blue"))
                            continue
                        test_lock = True
                        test_reward = training_thread.testing(
                            sess, args.eval_max_steps, temp_global_t,
                            summary_writer)
                        testing_rewards[temp_global_t] = test_reward
                        test_lock = False
                        last_temp_global_t = temp_global_t
                    finally:
                        lock.release()
                # all threads wait until evaluation finishes
                while not stop_requested and test_lock:
                    time.sleep(0.01)


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

    if not os.path.exists(folder):
        os.mkdir(folder)

    # write wall time
    wall_t = time.time() - start_time
    wall_t_fname = folder + '/' + 'wall_t.' + str(global_t)
    with open(wall_t_fname, 'w') as f:
        f.write(str(wall_t))
    with open(folder + '/pretrain_global_t', 'w') as f:
        f.write(str(pretrain_global_t))

    saver.save(sess, folder + '/' + '{}_checkpoint'.format(args.gym_env.replace('-', '_')), global_step = global_t)

    pickle.dump(testing_rewards, open(folder + '/' + args.gym_env.replace('-', '_') + '-rewards.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(training_rewards, open(folder + '/' + args.gym_env.replace('-', '_') + '-train-rewards.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    print (colored('Data saved!', 'green'))
