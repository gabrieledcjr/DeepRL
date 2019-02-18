#!/usr/bin/env python3
"""Asynchronous Advantage Actor-Critic (A3C).

Usage:
python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --parallel-size=16
    --initial-learn-rate=7e-4 --use-lstm --use-mnih-2015

python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --parallel-size=16
    --initial-learn-rate=7e-4 --use-lstm --use-mnih-2015 --use-transfer
    --not-transfer-fc2 --transfer-folder=<>

python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --parallel-size=16
    --initial-learn-rate=7e-4 --use-lstm --use-mnih-2015 --use-transfer
    --not-transfer-fc2 --transfer-folder=<> --load-pretrained-model
    --onevsall-mtl --pretrained-model-folder=<>
    --use-pretrained-model-as-advice --use-pretrained-model-as-reward-shaping
"""
import logging
import numpy as np
import os
import pathlib
import signal
import sys
import threading
import time

from a3c_training_thread import A3CTrainingThread
from common.game_state import GameState
from common.util import load_memory
from common.util import prepare_dir
from game_ac_network import GameACFFNetwork
from game_ac_network import GameACLSTMNetwork
from sil_memory import SILReplayMemory

logger = logging.getLogger("a3c")

try:
    import cPickle as pickle
except ImportError:
    import pickle


def run_a3c(args):
    """Run A3C experiment."""
    GYM_ENV_NAME = args.gym_env.replace('-', '_')

    if args.use_gpu:
        assert args.cuda_devices != ''
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import tensorflow as tf

    if not os.path.exists('results/a3c'):
        os.makedirs('results/a3c')

    if args.folder is not None:
        folder = 'results/a3c/{}_{}'.format(GYM_ENV_NAME, args.folder)
    else:
        folder = 'results/a3c/{}'.format(GYM_ENV_NAME)
        end_str = ''

        if args.use_mnih_2015:
            end_str += '_mnih2015'
        if args.padding == 'SAME':
            end_str += '_same'
        if args.use_lstm:
            end_str += '_lstm'
        if args.unclipped_reward:
            end_str += '_rawreward'
        elif args.log_scale_reward:
            end_str += '_logreward'
        if args.transformed_bellman:
            end_str += '_transformedbell'

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

        if args.load_pretrained_model:
            if args.use_pretrained_model_as_advice:
                end_str += '_modelasadvice'
            if args.use_pretrained_model_as_reward_shaping:
                end_str += '_modelasshaping'

        if args.use_sil:
            end_str += '_sil'

        folder += end_str

    if args.append_experiment_num is not None:
        folder += '_' + args.append_experiment_num

    folder = pathlib.Path(folder)

    demo_memory = None
    if args.load_memory or args.load_demo_cam:
        if args.demo_memory_folder is not None:
            demo_memory_folder = args.demo_memory_folder
        else:
            demo_memory_folder = 'collected_demo/{}'.format(GYM_ENV_NAME)
        demo_memory_folder = pathlib.Path(demo_memory_folder)

    if args.load_memory and args.use_sil:
        demo_memory, actions_ctr, total_rewards, total_steps = \
            load_memory(
                name=None,
                demo_memory_folder=demo_memory_folder,
                demo_ids=args.demo_ids,
                imgs_normalized=False)

    demo_memory_cam = None
    if args.load_demo_cam:
        demo_cam, _, total_rewards_cam, _ = load_memory(
            name=None,
            demo_memory_folder=demo_memory_folder,
            demo_ids=args.demo_cam_id,
            imgs_normalized=False)

        demo_cam = demo_cam[int(args.demo_cam_id)]
        demo_memory_cam = np.zeros(
            (len(demo_cam),
             demo_cam.height,
             demo_cam.width,
             demo_cam.phi_length),
            dtype=np.float32)
        for i in range(len(demo_cam)):
            s0 = (demo_cam[i])[0]
            demo_memory_cam[i] = np.copy(s0)
        del demo_cam
        logger.info("loaded demo {} for testing CAM".format(args.demo_cam_id))

    device = "/cpu:0"
    gpu_options = None
    if args.use_gpu:
        device = "/gpu:"+os.environ["CUDA_VISIBLE_DEVICES"]
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=args.gpu_fraction)

    initial_learning_rate = args.initial_learn_rate
    logger.info('Initial Learning Rate={}'.format(initial_learning_rate))
    time.sleep(2)

    global_t = 0
    pretrain_global_t = 0
    pretrain_epoch = 0
    rewards = {'train': {}, 'eval': {}}
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
            from game_class_network import MTLBinaryClassNetwork \
                as PretrainedModelNetwork
        else:
            from game_class_network import MultiClassNetwork \
                as PretrainedModelNetwork
            logger.error("Not supported yet!")
            assert False

        if args.pretrained_model_folder is not None:
            pretrained_model_folder = args.pretrained_model_folder
        else:
            pretrained_model_folder = GYM_ENV_NAME
            pretrained_model_folder += '_classifier_use_mnih_onevsall_mtl'

        pretrained_model_folder = pathlib.Path(pretrained_model_folder)

        PretrainedModelNetwork.use_mnih_2015 = args.use_mnih_2015
        pretrained_model = PretrainedModelNetwork(action_size, -1, device)
        pretrained_model_sess = tf.Session(config=config,
                                           graph=pretrained_model.graph)
        pretrained_model.load(
            pretrained_model_sess,
            pretrained_model_folder / '{}_checkpoint'.format(GYM_ENV_NAME))

    input_shape = (args.input_shape, args.input_shape, 4)
    if args.use_lstm:
        GameACLSTMNetwork.use_mnih_2015 = args.use_mnih_2015
        global_network = GameACLSTMNetwork(
            action_size, -1, device, padding=args.padding,
            in_shape=input_shape)
    else:
        GameACFFNetwork.use_mnih_2015 = args.use_mnih_2015
        global_network = GameACFFNetwork(
            action_size, -1, device, padding=args.padding,
            in_shape=input_shape)

    all_workers = []

    learning_rate_input = tf.placeholder(tf.float32, shape=(), name="opt_lr")

    grad_applier = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate_input,
        decay=args.rmsp_alpha,
        epsilon=args.rmsp_epsilon)

    A3CTrainingThread.log_interval = args.log_interval
    A3CTrainingThread.perf_log_interval = args.performance_log_interval
    A3CTrainingThread.local_t_max = args.local_t_max
    A3CTrainingThread.use_lstm = args.use_lstm
    A3CTrainingThread.action_size = action_size
    A3CTrainingThread.entropy_beta = args.entropy_beta
    A3CTrainingThread.gamma = args.gamma
    A3CTrainingThread.use_mnih_2015 = args.use_mnih_2015
    A3CTrainingThread.env_id = args.gym_env
    A3CTrainingThread.finetune_upper_layers_only = \
        args.finetune_upper_layers_only
    A3CTrainingThread.transformed_bellman = args.transformed_bellman
    A3CTrainingThread.clip_norm = args.grad_norm_clip
    A3CTrainingThread.use_grad_cam = args.use_grad_cam
    A3CTrainingThread.use_sil = args.use_sil
    A3CTrainingThread.log_idx = 1 if args.use_sil else 0

    if args.unclipped_reward:
        A3CTrainingThread.reward_type = "RAW"
    elif args.log_scale_reward:
        A3CTrainingThread.reward_type = "LOG"
    else:
        A3CTrainingThread.reward_type = "CLIP"

    shared_memory_sil = None
    if args.use_sil:
        shared_memory_sil = SILReplayMemory(
            action_size, max_len=250000, gamma=args.gamma,
            clip=False if args.unclipped_reward else True,
            height=input_shape[0], width=input_shape[1],
            phi_length=input_shape[2])
        shared_memory_sil.counter = 0

        if demo_memory is not None:
            temp_memory = SILReplayMemory(
                action_size, max_len=10**5, gamma=args.gamma,
                clip=False if args.unclipped_reward else True,
                height=input_shape[0], width=input_shape[1],
                phi_length=input_shape[2])

            for idx in list(demo_memory.keys()):
                demo = demo_memory[idx]
                for i in range(len(demo)+1):
                    s0, a0, _, _, _, r1, t1, _ = demo[i]
                    temp_memory.add_item(s0, a0, r1, t1)

                    if t1:  # terminal
                        shared_memory_sil.extend(temp_memory)

                if len(temp_memory) > 0:
                    logger.warning("Disregard {} states in"
                                   " demo_memory {}".format(
                                    len(temp_memory), idx))
                    temp_memory.reset()
            logger.info("SIL: memory size {}".format(len(shared_memory_sil)))

            del temp_memory

    n_shapers = args.parallel_size  # int(args.parallel_size * .25)
    mod = args.parallel_size // n_shapers
    for i in range(args.parallel_size):
        is_reward_shape = False
        is_advice = False
        sil_thread = i == 0 and args.use_sil
        if i % mod == 0:
            is_reward_shape = args.use_pretrained_model_as_reward_shaping
            is_advice = args.use_pretrained_model_as_advice

        if args.use_lstm:
            local_network = GameACLSTMNetwork(
                action_size, i, device, padding=args.padding,
                in_shape=input_shape)
        else:
            local_network = GameACFFNetwork(
                action_size, i, device, padding=args.padding,
                in_shape=input_shape)

        a3c_worker = A3CTrainingThread(
            i, global_network, local_network, initial_learning_rate,
            learning_rate_input,
            grad_applier, args.max_time_step,
            device=device,
            pretrained_model=pretrained_model,
            pretrained_model_sess=pretrained_model_sess,
            advice=is_advice,
            reward_shaping=is_reward_shape,
            sil_thread=sil_thread)

        all_workers.append(a3c_worker)

    # prepare session
    sess = tf.Session(config=config)

    if args.use_transfer:
        if args.transfer_folder is not None:
            transfer_folder = args.transfer_folder
        else:
            transfer_folder = 'results/pretrain_models/{}'.format(GYM_ENV_NAME)
            end_str = ''
            if args.use_mnih_2015:
                end_str += '_mnih2015'
            end_str += '_l2beta1E-04_batchprop'  # TODO: make this an argument
            transfer_folder += end_str

        transfer_folder = pathlib.Path(transfer_folder)
        transfer_folder /= 'transfer_model'

        if args.not_transfer_conv2:
            transfer_var_list = [
                global_network.W_conv1,
                global_network.b_conv1,
                ]

        elif (args.not_transfer_conv3 and args.use_mnih_2015):
            transfer_var_list = [
                global_network.W_conv1,
                global_network.b_conv1,
                global_network.W_conv2,
                global_network.b_conv2,
                ]

        elif args.not_transfer_fc1:
            transfer_var_list = [
                global_network.W_conv1,
                global_network.b_conv1,
                global_network.W_conv2,
                global_network.b_conv2,
                ]

            if args.use_mnih_2015:
                transfer_var_list += [
                    global_network.W_conv3,
                    global_network.b_conv3,
                    ]

        elif args.not_transfer_fc2:
            transfer_var_list = [
                global_network.W_conv1,
                global_network.b_conv1,
                global_network.W_conv2,
                global_network.b_conv2,
                global_network.W_fc1,
                global_network.b_fc1,
                ]

            if args.use_mnih_2015:
                transfer_var_list += [
                    global_network.W_conv3,
                    global_network.b_conv3,
                    ]

        else:
            transfer_var_list = [
                global_network.W_conv1,
                global_network.b_conv1,
                global_network.W_conv2,
                global_network.b_conv2,
                global_network.W_fc1,
                global_network.b_fc1,
                global_network.W_fc2,
                global_network.b_fc2,
                ]

            if args.use_mnih_2015:
                transfer_var_list += [
                    global_network.W_conv3,
                    global_network.b_conv3,
                    ]

            if '_sil' in str(transfer_folder):
                transfer_var_list += [
                    global_network.W_fc3,
                    global_network.b_fc3,
                    ]

        global_network.load_transfer_model(
            sess, folder=transfer_folder,
            not_transfer_fc2=args.not_transfer_fc2,
            not_transfer_fc1=args.not_transfer_fc1,
            not_transfer_conv3=(args.not_transfer_conv3
                                and args.use_mnih_2015),
            not_transfer_conv2=args.not_transfer_conv2,
            var_list=transfer_var_list,
            )

    def initialize_uninitialized(sess):
        global_vars = tf.global_variables()
        is_not_initialized = sess.run(
            [tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = \
            [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))

    if args.use_transfer:
        initialize_uninitialized(sess)
    else:
        sess.run(tf.global_variables_initializer())

    # summary writer for tensorboard
    # summary_op = tf.summary.merge_all()
    summ_file = 'results/log/a3c/{}/'.format(GYM_ENV_NAME) + str(folder)[12:]
    summary_writer = tf.summary.FileWriter(summ_file, sess.graph)

    # init or load checkpoint with saver
    root_saver = tf.train.Saver(max_to_keep=1)
    saver = tf.train.Saver(max_to_keep=6)
    best_saver = tf.train.Saver(max_to_keep=1)
    checkpoint = tf.train.get_checkpoint_state(str(folder))
    if checkpoint and checkpoint.model_checkpoint_path:
        root_saver.restore(sess, checkpoint.model_checkpoint_path)
        logger.info("checkpoint loaded:{}".format(
            checkpoint.model_checkpoint_path))
        tokens = checkpoint.model_checkpoint_path.split("-")
        # set global step
        global_t = int(tokens[-1])
        logger.info(">>> global step set: {}".format(global_t))

        # set wall time
        wall_t_fname = folder / 'wall_t.{}'.format(global_t)
        with wall_t_fname.open('r') as f:
            wall_t = float(f.read())

        pretrain_t_file = folder / 'pretrain_global_t'
        with pretrain_t_file.open('r') as f:
            pretrain_global_t = int(f.read())

        best_reward_file = folder / 'model_best/best_model_reward'
        with best_reward_file.open('r') as f:
            best_model_reward = float(f.read())
        reward_file = folder / '{}-a3c-rewards.pkl'.format(GYM_ENV_NAME)
        rewards = pickle.load(reward_file.open('rb'))
    else:
        logger.warning("Could not find old checkpoint")
        # set wall time
        wall_t = 0.0
        prepare_dir(folder, empty=True)
        prepare_dir(folder / 'model_checkpoints', empty=True)
        prepare_dir(folder / 'model_best', empty=True)
        prepare_dir(folder / 'frames', empty=True)

    lock = threading.Lock()
    ctr_lock = threading.Lock()
    sil_lock = None
    if args.use_sil:
        sil_lock = threading.Lock()

    test_lock = False
    if global_t == 0:
        test_lock = True

    def next_t(current_t, freq):
        return np.ceil((current_t + 0.00001) / freq) * freq

    next_global_t = next_t(global_t, args.eval_freq)
    next_save_t = next_t(
        global_t, (args.max_time_step * args.max_time_step_fraction) // 5)

    last_temp_global_t = global_t
    ispretrain_markers = [False] * args.parallel_size
    threads_ctr = args.parallel_size

    def train_function(parallel_idx):
        nonlocal global_t, pretrain_global_t, pretrain_epoch, \
            rewards, test_lock, lock, sil_lock, next_global_t, next_save_t, \
            threads_ctr, last_temp_global_t, ispretrain_markers, \
            shared_memory_sil

        a3c_worker = all_workers[parallel_idx]
        a3c_worker.set_summary_writer(summary_writer)

        # Evaluate model before training
        if not stop_requested and global_t == 0:
            with lock:
                if parallel_idx == 0:
                    rewards['eval'][global_t] = all_workers[0].testing(
                        sess, args.eval_max_steps, global_t, folder,
                        demo_memory_cam=demo_memory_cam)
                    checkpt_file = folder / 'model_checkpoints'
                    checkpt_file /= '{}_checkpoint'.format(GYM_ENV_NAME)
                    saver.save(sess, str(checkpt_file), global_step=global_t)
                    save_best_model(rewards['eval'][global_t][0])
                    test_lock = False

            # all threads wait until evaluation finishes
            while not stop_requested and test_lock:
                time.sleep(0.01)

        # set start_time
        start_time = time.time() - wall_t
        a3c_worker.set_start_time(start_time)
        episode_end = True
        sil_ctr = 0

        while True:
            if stop_requested:
                return

            if global_t >= (args.max_time_step * args.max_time_step_fraction):
                return

            if a3c_worker.sil_thread:
                # SIL
                with ctr_lock:
                    threads_ctr -= 1

                diff_global_t = 0
                # if shared_memory_sil.counter >= 4: #len(all_workers) - 1:
                #     with sil_lock:
                #         shared_memory_sil.counter = 0

                sil_ctr = a3c_worker.sil_train(
                    sess, global_t, shared_memory_sil, sil_lock, sil_ctr,
                    batch_size=512, m=4)

                with ctr_lock:
                    threads_ctr += 1

            else:
                with ctr_lock:
                    threads_ctr -= 1

                diff_global_t, episode_end, part_end = a3c_worker.train(
                    sess, global_t, rewards)

                with ctr_lock:
                    threads_ctr += 1

                if shared_memory_sil is not None:
                    with sil_lock:
                        # shared_memory_sil.counter += 1
                        if part_end:
                            shared_memory_sil.extend(a3c_worker.episode)

            with lock:
                global_t += diff_global_t

                if global_t > next_global_t:
                    next_global_t = next_t(global_t, args.eval_freq)

                    # wait for all threads to finish before testing
                    while not stop_requested and threads_ctr < len(all_workers):
                        time.sleep(0.01)

                    step_t = int(next_global_t - args.eval_freq)
                    rewards['eval'][step_t] = a3c_worker.testing(
                        sess, args.eval_max_steps, step_t, folder,
                        demo_memory_cam=demo_memory_cam)
                    save_best_model(rewards['eval'][step_t][0])

            if global_t > next_save_t:
                freq = (args.max_time_step * args.max_time_step_fraction) // 5
                next_save_t = next_t(global_t, freq)
                checkpt_file = folder / 'model_checkpoints'
                checkpt_file /= '{}_checkpoint'.format(GYM_ENV_NAME)
                saver.save(sess, str(checkpt_file), global_step=global_t,
                           write_meta_graph=False)

    def signal_handler(signal, frame):
        nonlocal stop_requested
        logger.info('You pressed Ctrl+C!')
        stop_requested = True

        if stop_requested and global_t == 0:
            sys.exit(1)

    def save_best_model(test_reward):
        nonlocal best_model_reward
        if test_reward > best_model_reward:
            best_model_reward = test_reward
            best_reward_file = folder / 'model_best/best_model_reward'

            with best_reward_file.open('w') as f:
                f.write(str(best_model_reward))

            best_checkpt_file = folder / 'model_best'
            best_checkpt_file /= '{}_checkpoint'.format(GYM_ENV_NAME)
            best_saver.save(sess, str(best_checkpt_file))

    train_threads = []
    for i in range(args.parallel_size):
        worker_thread = threading.Thread(target=train_function, args=(i,))
        train_threads.append(worker_thread)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # set start time
    start_time = time.time() - wall_t

    for t in train_threads:
        t.start()

    print('Press Ctrl+C to stop')

    for t in train_threads:
        t.join()

    logger.info('Now saving data. Please wait')

    # write wall time
    wall_t = time.time() - start_time
    wall_t_fname = folder / 'wall_t.{}'.format(global_t)
    with wall_t_fname.open('w') as f:
        f.write(str(wall_t))

    pretrain_gt_fname = folder / 'pretrain_global_t'
    with pretrain_gt_fname.open('w') as f:
        f.write(str(pretrain_global_t))

    checkpoint_file = str(folder / '{}_checkpoint_a3c'.format(GYM_ENV_NAME))
    root_saver.save(sess, checkpoint_file, global_step=global_t)

    reward_fname = folder / '{}-a3c-rewards.pkl'.format(GYM_ENV_NAME)
    pickle.dump(rewards, reward_fname.open('wb'), pickle.HIGHEST_PROTOCOL)
    logger.info('Data saved!')

    sess.close()
