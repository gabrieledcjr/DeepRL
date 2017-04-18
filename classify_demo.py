# -*- coding: utf-8 -*-
import os
import time

from game_class_network import GameACFFNetwork, GameACLSTMNetwork


class ClassifyDemo(object):
    def __init__(self):
        pass

    def _load_memory(self):
        pass

    def run(self):
        data = {
            'training_step': [],
            'training_accuracy': [],
            'training_entropy': [],
            'testing_step': [],
            'testing_accuracy': [],
            'testing_entropy': [],
            'max_accuracy': 0.,
            'max_accuracy_step': 0,
        }

def classify_demo(args):
    if args.use_gpu:
        assert args.cuda_devices != ''
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import tensorflow as tf

    device = "/cpu:0"
    gpu_options = None
    if args.use_gpu:
        device = "/gpu:"+os.environ["CUDA_VISIBLE_DEVICES"]
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)

    assert args.initial_learn_rate > 0
    initial_learning_rate = args.initial_learn_rate
    print (colored('Initial Learning Rate={}'.format(initial_learning_rate), 'green'))
    time.sleep(2)

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

    grad_applier = RMSPropApplier(
        learning_rate = initial_learning_rate,
        decay = args.rmsp_alpha,
        momentum = 0.0,
        epsilon = args.rmsp_epsilon,
        clip_norm = args.grad_norm_clip,
        device = device)

    # TODO Training code here

    # prepare session
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
        allow_soft_placement=True)
    sess = tf.Session(config=config)

    init = tf.global_variables_initializer()
    sess.run(init)
