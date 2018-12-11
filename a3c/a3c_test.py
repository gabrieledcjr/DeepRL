#!/usr/bin/env python3
import threading
import numpy as np
import signal
import math
import os
import sys
import time
import logging

from common.replay_memory import ReplayMemory
from common.util import load_memory, prepare_dir
from common.game_state import GameState

logger = logging.getLogger("a3c")

try:
    import cPickle as pickle
except ImportError:
    import pickle

def run_a3c_test(args):
    """
    python3 run_experiment.py --gym-env=PongNoFrameskip-v4 --test-model --initial-learn-rate=7e-4 --use-mnih-2015 --folder=<> --eval-max-steps=5000 --use-grad-cam
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
        folder = args.folder
    else:
        folder = 'results/a3c/{}'.format(args.gym_env.replace('-', '_'))
        end_str = ''

        if args.use_mnih_2015:
            end_str += '_mnih2015'
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
        if args.train_with_demo_num_steps > 0 or args.train_with_demo_num_epochs > 0:
            end_str += '_pretrain_ina3c'
        if args.use_demo_threads:
            end_str += '_demothreads'

        if args.load_pretrained_model:
            if args.use_pretrained_model_as_advice:
                end_str += '_modelasadvice'
            if args.use_pretrained_model_as_reward_shaping:
                end_str += '_modelasshaping'
        folder += end_str

    demo_memory_cam = None
    demo_cam_human = False
    if args.load_demo_cam:
        if args.demo_memory_folder is not None:
            demo_memory_folder = args.demo_memory_folder
        else:
            demo_memory_folder = 'collected_demo/{}'.format(args.gym_env.replace('-', '_'))

        if args.demo_cam_id is not None:
            demo_cam_human = True
            demo_cam, _, total_rewards_cam, _ = load_memory(
                name=None,
                demo_memory_folder=demo_memory_folder,
                demo_ids=args.demo_cam_id,
                imgs_normalized=False)

            demo_cam = demo_cam[int(args.demo_cam_id)]
            logger.info("loaded demo {} for testing CAM".format(args.demo_cam_id))
        else:
            demo_cam = ReplayMemory()
            demo_cam.load(name='test_cam', folder=args.demo_cam_folder)
            logger.info("loaded demo {} for testing CAM".format(args.demo_cam_folder + '/test_cam'))

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

    device = "/cpu:0"
    gpu_options = None
    if args.use_gpu:
        device = "/gpu:"+os.environ["CUDA_VISIBLE_DEVICES"]
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)

    initial_learning_rate = args.initial_learn_rate
    logger.info('Initial Learning Rate={}'.format(initial_learning_rate))
    time.sleep(2)

    global_t = 0
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

    if args.use_lstm:
        GameACLSTMNetwork.use_mnih_2015 = args.use_mnih_2015
        global_network = GameACLSTMNetwork(action_size, -1, device)
    else:
        GameACFFNetwork.use_mnih_2015 = args.use_mnih_2015
        global_network = GameACFFNetwork(action_size, -1, device)

    training_threads = []

    learning_rate_input = tf.placeholder(tf.float32, shape=(), name="opt_lr")

    grad_applier = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate_input,
        decay=args.rmsp_alpha,
        epsilon=args.rmsp_epsilon)

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
    A3CTrainingThread.finetune_upper_layers_only = args.finetune_upper_layers_only
    A3CTrainingThread.transformed_bellman = args.transformed_bellman
    A3CTrainingThread.clip_norm = args.grad_norm_clip
    A3CTrainingThread.use_grad_cam = args.use_grad_cam

    if args.unclipped_reward:
        A3CTrainingThread.reward_type = "RAW"
    elif args.log_scale_reward:
        A3CTrainingThread.reward_type = "LOG"
    else:
        A3CTrainingThread.reward_type = "CLIP"

    testing_thread = A3CTrainingThread(
        0, global_network, initial_learning_rate,
        learning_rate_input,
        grad_applier, 0,
        device=device)

    # prepare session
    sess = tf.Session(config=config)

    if args.use_transfer:
        if args.transfer_folder is not None:
            transfer_folder = args.transfer_folder
        else:
            transfer_folder = 'results/pretrain_models/{}'.format(args.gym_env.replace('-', '_'))
            end_str = ''
            if args.use_mnih_2015:
                end_str += '_mnih2015'
            end_str += '_l2beta1E-04_batchprop'  #TODO: make this an argument
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

    # init or load checkpoint with saver
    root_saver = tf.train.Saver(max_to_keep=1)
    checkpoint = tf.train.get_checkpoint_state(folder)
    if checkpoint and checkpoint.model_checkpoint_path:
        root_saver.restore(sess, checkpoint.model_checkpoint_path)
        logger.info("checkpoint loaded:{}".format(checkpoint.model_checkpoint_path))
        tokens = checkpoint.model_checkpoint_path.split("-")
        # set global step
        global_t = int(tokens[-1])
        logger.info(">>> global step set: {}".format(global_t))
    else:
        logger.warning("Could not find old checkpoint")

    def test_function():
        nonlocal global_t

        if args.use_transfer:
            from_folder = transfer_folder.split('/')[-2]
        else:
            from_folder = folder.split('/')[-1]

        save_folder = 'results/test_model/a3c/{}'.format(from_folder)
        prepare_dir(save_folder, empty=False)
        prepare_dir(save_folder + '/frames', empty=False)

        # Evaluate model before training
        if not stop_requested:
            testing_thread.testing_model(
                sess, args.eval_max_steps, global_t, save_folder,
                demo_memory_cam=demo_memory_cam, demo_cam_human=demo_cam_human)

    def signal_handler(signal, frame):
        nonlocal stop_requested
        logger.info('You pressed Ctrl+C!')
        stop_requested = True

        if stop_requested and global_t == 0:
            sys.exit(1)

    test_thread = threading.Thread(target=test_function, args=())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    test_thread.start()

    print ('Press Ctrl+C to stop')

    test_thread.join()

    sess.close()
