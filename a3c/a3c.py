#!/usr/bin/env python3
import threading
import numpy as np
import signal
import math
import os
import sys
import time
import logging

from common.util import load_memory, prepare_dir
from common.game_state import GameState
from rmsprop_applier import RMSPropApplier

logger = logging.getLogger("a3c")

try:
    import cPickle as pickle
except ImportError:
    import pickle

def run_a3c(args):
    """
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --parallel-size=16 --initial-learn-rate=7e-4 --use-lstm --use-mnih-2015

    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --parallel-size=16 --initial-learn-rate=7e-4 --use-lstm --use-mnih-2015 --use-transfer --not-transfer-fc2 --transfer-folder=<>

    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --parallel-size=16 --initial-learn-rate=7e-4 --use-lstm --use-mnih-2015 --use-transfer --not-transfer-fc2 --transfer-folder=<> --load-pretrained-model --onevsall-mtl --pretrained-model-folder=<> --use-pretrained-model-as-advice --use-pretrained-model-as-reward-shaping
    """
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

    if not os.path.exists('results/a3c'):
        os.makedirs('results/a3c')

    if args.folder is not None:
        folder = 'results/a3c/{}_{}'.format(args.gym_env.replace('-', '_'), args.folder)
    else:
        folder = 'results/a3c/{}'.format(args.gym_env.replace('-', '_'))
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
        if args.finetune_upper_layers_only:
            end_str += '_tune_upperlayers'
        if args.train_with_demo_num_steps > 0 or args.train_with_demo_num_epochs > 0:
            end_str += '_pretrain_ina3c'
        if args.use_demo_threads:
            end_str += '_demothreads'
        if args.use_mnih_2015:
            end_str += '_mnih2015'
        if args.use_lstm:
            end_str += '_lstm'
        if args.log_scale_reward:
            end_str += '_log_reward'
        if args.use_egreedy_threads:
            end_str += '_egreedy'
        if args.load_pretrained_model:
            if args.use_pretrained_model_as_advice:
                end_str += '_modelasadvice'
            if args.use_pretrained_model_as_reward_shaping:
                end_str += '_modelasshaping'
        folder += end_str

    if args.append_experiment_num is not None:
        folder += '_' + args.append_experiment_num

    if False:
        from log_formatter import LogFormatter
        fh = logging.FileHandler('{}/a3c.log'.format(folder), mode='w')
        fh.setLevel(logging.DEBUG)
        formatter = LogFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

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
    logger.info('Initial Learning Rate={}'.format(initial_learning_rate))
    time.sleep(2)

    global_t = 0
    pretrain_global_t = 0
    pretrain_epoch = 0
    rewards = {'train':{}, 'eval':{}}
    best_model_reward = -(sys.maxsize)

    stop_requested = False

    game_state = GameState(env_id=args.gym_env)
    action_size = game_state.env.action_space.n
    game_state.close()
    del game_state.env
    del game_state

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
        allow_soft_placement=True)

    pretrained_model = None
    pretrained_model_sess = None
    if args.load_pretrained_model:
        if args.onevsall_mtl:
            from game_class_network import MTLBinaryClassNetwork as PretrainedModelNetwork
        elif args.onevsall_mtl_linear:
            from game_class_network import MTLMultivariateNetwork as PretrainedModelNetwork
        else:
            from game_class_network import MultiClassNetwork as PretrainedModelNetwork
            logger.error("Not supported yet!")
            assert False

        if args.pretrained_model_folder is not None:
            pretrained_model_folder = args.pretrained_model_folder
        else:
            pretrained_model_folder = '{}_classifier_use_mnih_onevsall_mtl'.format(args.gym_env.replace('-', '_'))
        PretrainedModelNetwork.use_mnih_2015 = args.use_mnih_2015
        pretrained_model = PretrainedModelNetwork(action_size, -1, device)
        pretrained_model_sess = tf.Session(config=config, graph=pretrained_model.graph)
        pretrained_model.load(pretrained_model_sess, '{}/{}_checkpoint'.format(pretrained_model_folder, args.gym_env.replace('-', '_')))

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
    A3CTrainingThread.egreedy_testing = args.egreedy_testing
    A3CTrainingThread.log_scale_reward = args.log_scale_reward
    A3CTrainingThread.finetune_upper_layers_only = args.finetune_upper_layers_only
    n_shapers = args.parallel_size #int(args.parallel_size * .25)
    mod = args.parallel_size // n_shapers
    for i in range(args.parallel_size):
        is_reward_shape = False
        is_advice = False
        if i % mod == 0:
            is_reward_shape = args.use_pretrained_model_as_reward_shaping
            is_advice = args.use_pretrained_model_as_advice
        training_thread = A3CTrainingThread(
            i, global_network, initial_learning_rate,
            learning_rate_input,
            grad_applier, args.max_time_step,
            device=device,
            pretrained_model=pretrained_model,
            pretrained_model_sess=pretrained_model_sess,
            advice=is_advice,
            reward_shaping=is_reward_shape)
        training_threads.append(training_thread)

    # prepare session
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

        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))

    if args.use_transfer:
        initialize_uninitialized(sess)
    else:
        sess.run(tf.global_variables_initializer())

    # summary for tensorboard
    score_input = tf.placeholder(tf.float32)
    tf.summary.scalar("score", score_input)
    steps_input = tf.placeholder(tf.int32)
    tf.summary.scalar("steps", steps_input)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('results/log/a3c/{}/'.format(args.gym_env.replace('-', '_')) + folder[12:], sess.graph)

    # init or load checkpoint with saver
    root_saver = tf.train.Saver(max_to_keep=1)
    saver = tf.train.Saver(max_to_keep=6)
    best_saver = tf.train.Saver(max_to_keep=1)
    checkpoint = tf.train.get_checkpoint_state(folder)
    if checkpoint and checkpoint.model_checkpoint_path:
        root_saver.restore(sess, checkpoint.model_checkpoint_path)
        logger.info("checkpoint loaded:{}".format(checkpoint.model_checkpoint_path))
        tokens = checkpoint.model_checkpoint_path.split("-")
        # set global step
        global_t = int(tokens[1])
        logger.info(">>> global step set: {}".format(global_t))
        # set wall time
        wall_t_fname = folder + '/' + 'wall_t.' + str(global_t)
        with open(wall_t_fname, 'r') as f:
            wall_t = float(f.read())
        with open(folder + '/pretrain_global_t', 'r') as f:
            pretrain_global_t = int(f.read())
        with open(folder + '/model_best/best_model_reward', 'r') as f_best_model_reward:
            best_model_reward = float(f_best_model_reward.read())
        rewards = pickle.load(open(folder + '/' + args.gym_env.replace('-', '_') + '-a3c-rewards.pkl', 'rb'))
    else:
        logger.warning("Could not find old checkpoint")
        # set wall time
        wall_t = 0.0
        prepare_dir(folder, empty=True)
        prepare_dir(folder + '/model_checkpoints', empty=True)
        prepare_dir(folder + '/model_best', empty=True)

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
            rewards, test_lock, lock, \
            last_temp_global_t, ispretrain_markers, num_demo_thread, \
            ctr_demo_thread
        training_thread = training_threads[parallel_index]

        # set all threads as demo threads
        training_thread.is_demo_thread = args.load_memory and args.use_demo_threads
        if training_thread.is_demo_thread or args.train_with_demo_num_steps > 0 or args.train_with_demo_num_epochs:
            training_thread.pretrain_init(demo_memory)

        if args.use_egreedy_threads and parallel_index >= args.parallel_size - args.parallel_size/2:
            logger.info("t_idx={} set as egreedy thread".format(parallel_index))
            training_thread.is_egreedy = True

        if global_t == 0 and (args.train_with_demo_num_steps > 0 or args.train_with_demo_num_epochs > 0) and parallel_index < 2:
            ispretrain_markers[parallel_index] = True
            training_thread.replay_mem_reset()

            # Pretraining with demo memory
            logger.info("t_idx={} pretrain starting".format(parallel_index))
            while ispretrain_markers[parallel_index]:
                if stop_requested:
                    return
                if pretrain_global_t > args.train_with_demo_num_steps and pretrain_epoch > args.train_with_demo_num_epochs:
                    # At end of pretraining, reset state
                    training_thread.replay_mem_reset()
                    training_thread.episode_reward = 0
                    training_thread.local_t = 0
                    if args.use_lstm:
                        training_thread.local_network.reset_state()
                    ispretrain_markers[parallel_index] = False
                    logger.info("t_idx={} pretrain ended".format(parallel_index))
                    break

                diff_pretrain_global_t, _ = training_thread.demo_process(
                    sess, pretrain_global_t)
                for _ in range(diff_pretrain_global_t):
                    pretrain_global_t += 1
                    if pretrain_global_t % 10000 == 0:
                        logger.debug("pretrain_global_t={}".format(pretrain_global_t))

                pretrain_epoch += 1
                if pretrain_epoch % 1000 == 0:
                    logger.debug("pretrain_epoch={}".format(pretrain_epoch))

            # Waits for all threads to finish pretraining
            while not stop_requested and any(ispretrain_markers):
                time.sleep(0.01)

        # Evaluate model before training
        if not stop_requested and global_t == 0:
            with lock:
                if parallel_index == 0:
                    test_reward, test_steps = training_threads[0].testing(
                        sess, args.eval_max_steps, global_t,
                        summary_writer)
                    rewards['eval'][global_t] = (test_reward, test_steps)
                    saver.save(sess, folder + '/model_checkpoints/' + '{}_checkpoint'.format(args.gym_env.replace('-', '_')), global_step=global_t)
                    save_best_model(test_reward)
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
                return
            if global_t > (args.max_time_step * args.max_time_step_fraction):
                return

            if args.use_demo_threads and global_t < args.max_steps_threads_as_demo and episode_end and num_demo_thread < 16:
                #if num_demo_thread < 2:
                demo_rate = 1.0 * (args.max_steps_threads_as_demo - global_t) / args.max_steps_threads_as_demo
                if demo_rate < 0.0333:
                    demo_rate = 0.0333

                if np.random.random() <= demo_rate and num_demo_thread < 16:
                    ctr_demo_thread += 1
                    training_thread.replay_mem_reset(D_idx=ctr_demo_thread%num_demos)
                    num_demo_thread += 1
                    logger.info("idx={} as demo thread started ({}/16) rate={}".format(parallel_index, num_demo_thread, demo_rate))
                    use_demo_thread = True

            if use_demo_thread:
                diff_global_t, episode_end = training_thread.demo_process(
                    sess, global_t)
                if episode_end:
                    num_demo_thread -= 1
                    use_demo_thread = False
                    logger.info("idx={} demo thread concluded ({}/16)".format(parallel_index, num_demo_thread))
            else:
                diff_global_t, episode_end = training_thread.process(
                    sess, global_t, summary_writer,
                    summary_op, score_input, steps_input, rewards)

            for _ in range(diff_global_t):
                global_t += 1
                if global_t % args.eval_freq == 0:
                    temp_global_t = global_t
                    lock.acquire()
                    try:
                        # catch multiple threads getting in at the same time
                        if last_temp_global_t == temp_global_t:
                            logger.info("Threading race problem averted!")
                            continue
                        test_lock = True
                        test_reward, test_steps = training_thread.testing(
                            sess, args.eval_max_steps, temp_global_t,
                            summary_writer)
                        rewards['eval'][temp_global_t] = (test_reward, test_steps)
                        if temp_global_t % ((args.max_time_step * args.max_time_step_fraction) // 5) == 0:
                            saver.save(
                                sess, folder + '/model_checkpoints/' + '{}_checkpoint'.format(args.gym_env.replace('-', '_')),
                                global_step=temp_global_t, write_meta_graph=False)
                        if test_reward > best_model_reward:
                            save_best_model(test_reward)
                        test_lock = False
                        last_temp_global_t = temp_global_t
                    finally:
                        lock.release()
                if global_t % ((args.max_time_step * args.max_time_step_fraction) // 5) == 0:
                    saver.save(
                        sess, folder + '/model_checkpoints/' + '{}_checkpoint'.format(args.gym_env.replace('-', '_')),
                        global_step=global_t, write_meta_graph=False)
                # all threads wait until evaluation finishes
                while not stop_requested and test_lock:
                    time.sleep(0.01)


    def signal_handler(signal, frame):
        nonlocal stop_requested
        logger.info('You pressed Ctrl+C!')
        stop_requested = True

        if stop_requested and global_t == 0:
            sys.exit(1)

    def save_best_model(test_reward):
        nonlocal best_model_reward
        best_model_reward = test_reward
        with open(folder + '/model_best/best_model_reward', 'w') as f_best_model_reward:
            f_best_model_reward.write(str(best_model_reward))
        best_saver.save(sess, folder + '/model_best/' + '{}_checkpoint'.format(args.gym_env.replace('-', '_')))

    train_threads = []
    for i in range(args.parallel_size):
        train_threads.append(threading.Thread(target=train_function, args=(i,)))

    signal.signal(signal.SIGINT, signal_handler)

    # set start time
    start_time = time.time() - wall_t

    for t in train_threads:
        t.start()

    print ('Press Ctrl+C to stop')
    signal.pause()

    logger.info('Now saving data. Please wait')

    for t in train_threads:
        t.join()

    # write wall time
    wall_t = time.time() - start_time
    wall_t_fname = folder + '/' + 'wall_t.' + str(global_t)
    with open(wall_t_fname, 'w') as f:
        f.write(str(wall_t))
    with open(folder + '/pretrain_global_t', 'w') as f:
        f.write(str(pretrain_global_t))

    root_saver.save(sess, folder + '/{}_checkpoint_a3c'.format(args.gym_env.replace('-', '_')), global_step=global_t)

    pickle.dump(rewards, open(folder + '/' + args.gym_env.replace('-', '_') + '-a3c-rewards.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    logger.info('Data saved!')

    sess.close()
