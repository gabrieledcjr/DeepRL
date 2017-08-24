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
from rmsprop_applier import RMSPropApplier

try:
    import cPickle as pickle
except ImportError:
    import pickle

def run_a3c(args):
    """
    python3 run_experiment.py --gym-env=PongDeterministic-v3 --parallel-size=16 --initial-learn-rate=7e-4 --use-lstm --use-mnih-2015
    """
    if args.not_gae:
        from game_ac_network_new import GameACFFNetwork, GameACLSTMNetwork
        from a3c_training_thread_new import A3CTrainingThread
    else:
        from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
        from a3c_training_thread import A3CTrainingThread
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
            if args.not_transfer_conv2:
                end_str += '_noconv2'
            elif args.not_transfer_conv3 and args.use_mnih_2015:
                end_str += '_noconv3'
            elif args.not_transfer_fc1:
                end_str += '_nofc1'
            elif args.not_transfer_fc2:
                end_str += '_nofc2'
        if args.train_with_demo_num_steps > 0 or args.train_with_demo_num_epochs > 0:
            end_str += '_pretrain_ina3c'
        if args.use_demo_threads:
            end_str += '_demothreads'
        if args.use_mnih_2015:
            end_str += '_use_mnih'
        if args.use_lstm:
            end_str += '_use_lstm'
        if args.adaptive_reward:
            end_str += '_adapt_reward'
        if args.auto_start:
            end_str += '_autostart'
        if args.not_gae:
            end_str += '_notgae'
        if args.use_egreedy_threads:
            end_str += '_use_egreedy'
        folder += end_str

    if args.append_experiment_num is not None:
        folder += '_' + args.append_experiment_num

    demo_memory = None
    num_demos = 0
    max_reward = 0.
    if args.load_memory:
        if args.demo_memory_folder is not None:
            demo_memory_folder = 'demo_samples/{}'.format(args.demo_memory_folder)
        else:
            demo_memory_folder = 'demo_samples/{}'.format(args.gym_env.replace('-', '_'))
        demo_memory, actions_ctr, max_reward = load_memory(args.gym_env, demo_memory_folder, imgs_normalized=True) #, create_symmetry=True)
        action_freq = [ actions_ctr[a] for a in range(demo_memory[0].num_actions) ]
        num_demos = len(demo_memory)

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
    if args.auto_start:
        action_size -= 1
    game_state.env.close()
    del game_state.env
    del game_state

    if args.use_lstm:
        GameACLSTMNetwork.use_mnih_2015 = args.use_mnih_2015
        GameACLSTMNetwork.l1_beta = args.l1_beta
        GameACLSTMNetwork.l2_beta = args.l2_beta
        global_network = GameACLSTMNetwork(action_size, -1, device)
    else:
        GameACFFNetwork.use_mnih_2015 = args.use_mnih_2015
        GameACFFNetwork.l1_beta = args.l1_beta
        GameACFFNetwork.l2_beta = args.l2_beta
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
    grad_applier_v = None
    if args.not_gae:
        grad_applier_v = RMSPropApplier(
            learning_rate = learning_rate_input,
            decay = args.rmsp_alpha,
            momentum = 0.0,
            epsilon = args.rmsp_epsilon,
            clip_norm = args.grad_norm_clip,
            device = device)

    A3CTrainingThread.log_interval = args.log_interval
    A3CTrainingThread.performance_log_interval = args.performance_log_interval
    A3CTrainingThread.local_t_max = args.local_t_max
    A3CTrainingThread.demo_t_max = args.demo_t_max
    A3CTrainingThread.use_lstm = args.use_lstm
    A3CTrainingThread.action_size = action_size
    A3CTrainingThread.entropy_beta = args.entropy_beta
    A3CTrainingThread.demo_entropy_beta = args.demo_entropy_beta
    A3CTrainingThread.gamma = args.gamma
    A3CTrainingThread.use_mnih_2015 = args.use_mnih_2015
    A3CTrainingThread.env_id = args.gym_env
    A3CTrainingThread.auto_start = args.auto_start
    A3CTrainingThread.egreedy_testing = args.egreedy_testing
    if args.adaptive_reward:
        A3CTrainingThread.adaptive_reward = True
        A3CTrainingThread.max_reward[0] = max_reward
    for i in range(args.parallel_size):
        training_thread = A3CTrainingThread(
            i, global_network, initial_learning_rate,
            learning_rate_input,
            grad_applier, args.max_time_step,
            device=device, grad_applier_v=grad_applier_v)
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

        if args.not_transfer_conv2:
            transfer_var_list = [global_network.W_conv1, global_network.b_conv1]
        elif (args.not_transfer_conv3 and args.use_mnih_2015):
            transfer_var_list = [
                global_network.W_conv1, global_network.b_conv1,
                global_network.W_conv2, global_network.b_conv2
            ]
        elif args.not_transfer_fc1:
            transfer_var_list = [
                global_network.W_conv1, global_network.b_conv1,
                global_network.W_conv2, global_network.b_conv2,
            ]
            if args.use_mnih_2015:
                transfer_var_list += [
                    global_network.W_conv3, global_network.b_conv3
                ]
        elif args.not_transfer_fc2:
            transfer_var_list = [
                global_network.W_conv1, global_network.b_conv1,
                global_network.W_conv2, global_network.b_conv2,
                global_network.W_fc1, global_network.b_fc1
            ]
            if args.use_mnih_2015:
                transfer_var_list += [
                    global_network.W_conv3, global_network.b_conv3
                ]
        else:
            transfer_var_list = [
                global_network.W_conv1, global_network.b_conv1,
                global_network.W_conv2, global_network.b_conv2,
                global_network.W_fc1, global_network.b_fc1,
                global_network.W_fc2, global_network.b_fc2
            ]
            if args.use_mnih_2015:
                transfer_var_list += [
                    global_network.W_conv3, global_network.b_conv3
                ]

        global_network.load_transfer_model(
            sess, folder=transfer_folder,
            not_transfer_fc2=args.not_transfer_fc2,
            not_transfer_fc1=args.not_transfer_fc1,
            not_transfer_conv3=(args.not_transfer_conv3 and args.use_mnih_2015),
            not_transfer_conv2=args.not_transfer_conv2,
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
    num_demo_thread = 0
    ctr_demo_thread = 0
    def train_function(parallel_index):
        nonlocal global_t, pretrain_global_t, pretrain_epoch, \
            testing_rewards, training_rewards, test_lock, lock, \
            last_temp_global_t, ispretrain_markers, num_demo_thread, ctr_demo_thread
        training_thread = training_threads[parallel_index]

        # set all threads as demo threads
        training_thread.is_demo_thread = args.load_memory and args.use_demo_threads
        if training_thread.is_demo_thread or args.train_with_demo_num_steps > 0 or args.train_with_demo_num_epochs:
            training_thread.pretrain_init(demo_memory)

        if args.use_egreedy_threads and parallel_index >= args.parallel_size - args.parallel_size/2:
            print ("t_idx={} set as egreedy thread".format(parallel_index))
            training_thread.is_egreedy = True

        if args.use_dropout:
            training_thread.use_dropout = True
            training_thread.keep_prob = 0.1

        if global_t == 0 and (args.train_with_demo_num_steps > 0 or args.train_with_demo_num_epochs > 0) and parallel_index < 2:
            ispretrain_markers[parallel_index] = True
            training_thread.replay_mem_reset()

            # Pretraining with demo memory
            print ("t_idx={} pretrain starting".format(parallel_index))
            while ispretrain_markers[parallel_index]:
                if stop_requested:
                    break
                if pretrain_global_t > args.train_with_demo_num_steps and pretrain_epoch > args.train_with_demo_num_epochs:
                    # At end of pretraining, reset state
                    training_thread.replay_mem_reset()
                    training_thread.episode_reward = 0
                    training_thread.local_t = 0
                    if args.use_lstm:
                        training_thread.local_network.reset_state()
                    ispretrain_markers[parallel_index] = False
                    print ("t_idx={} pretrain ended".format(parallel_index))
                    break

                diff_pretrain_global_t, _ = training_thread.demo_process(
                    sess, pretrain_global_t)
                for _ in range(diff_pretrain_global_t):
                    pretrain_global_t += 1
                    if pretrain_global_t % 10000 == 0:
                        print ("pretrain_global_t={}".format(pretrain_global_t))

                pretrain_epoch += 1
                if pretrain_epoch % 1000 == 0:
                    print ("pretrain_epoch={}".format(pretrain_epoch))

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

        # set start_time
        start_time = time.time() - wall_t
        training_thread.set_start_time(start_time)
        episode_end = True
        use_demo_thread = False
        while True:
            if stop_requested:
                break
            if global_t > (args.max_time_step * args.max_time_step_fraction):
                break

            if args.use_demo_threads and global_t < args.max_steps_threads_as_demo and episode_end and num_demo_thread < 16:
                #if num_demo_thread < 2:
                demo_rate = 1.0 * (args.max_steps_threads_as_demo - global_t) / args.max_steps_threads_as_demo
                if demo_rate < 0.0333:
                    demo_rate = 0.0333

                if np.random.random() <= demo_rate and num_demo_thread < 16:
                    ctr_demo_thread += 1
                    training_thread.replay_mem_reset(D_idx=ctr_demo_thread%num_demos)
                    num_demo_thread += 1
                    print (colored("idx={} as demo thread started ({}/16) rate={}".format(parallel_index, num_demo_thread, demo_rate), "yellow"))
                    use_demo_thread = True

            if use_demo_thread:
                diff_global_t, episode_end = training_thread.demo_process(
                    sess, global_t)
                if episode_end:
                    num_demo_thread -= 1
                    use_demo_thread = False
                    print (colored("idx={} demo thread concluded ({}/16)".format(parallel_index, num_demo_thread), "green"))
            else:
                diff_global_t, episode_end = training_thread.process(
                    sess, global_t, summary_writer,
                    summary_op, score_input, training_rewards)

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
