# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import sys

from game_class_network import MultiClassNetwork
from util import load_memory
from game_state import GameState
from termcolor import colored


def extract_layers(args):
    '''
    python3 run_experiment.py --gym-env=PongDeterministic-v3 --extract-transfer-layers --use-mnih-2015
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import tensorflow as tf

    device = "/cpu:0"

    if args.model_folder is not None:
        model_folder = '{}_{}'.format(args.gym_env.replace('-', '_'), args.model_folder)
    else:
        model_folder = '{}_classifier'.format(args.gym_env.replace('-', '_'))
        end_str = ''
        if args.use_mnih_2015:
            end_str += '_use_mnih'
        if args.use_lstm:
            end_str += '_use_lstm'
        model_folder += end_str

    print ("Model folder:{}".format(model_folder))

    if not os.path.exists(model_folder + '/transfer_model'):
        os.makedirs(model_folder + '/transfer_model')
    if not os.path.exists(model_folder + '/transfer_model/all'):
        os.makedirs(model_folder + '/transfer_model/all')
    if not os.path.exists(model_folder + '/transfer_model/nofc2'):
        os.makedirs(model_folder + '/transfer_model/nofc2')
    if not os.path.exists(model_folder + '/transfer_model/nofc1'):
        os.makedirs(model_folder + '/transfer_model/nofc1')
    if args.use_mnih_2015 and not os.path.exists(model_folder + '/transfer_model/noconv3'):
        os.makedirs(model_folder + '/transfer_model/noconv3')
    if not os.path.exists(model_folder + '/transfer_model/noconv2'):
        os.makedirs(model_folder + '/transfer_model/noconv2')

    game_state = GameState(env_id=args.gym_env)
    action_size = game_state.env.n_actions
    game_state.env.close()
    del game_state.env
    del game_state

    MultiClassNetwork.use_mnih_2015 = args.use_mnih_2015
    network = MultiClassNetwork(action_size, -1, device)

    with tf.device(device):
        opt = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=0.001)

    # prepare session
    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

    # init or load checkpoint with saver
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print(colored("checkpoint loaded:{}".format(checkpoint.model_checkpoint_path), "green"))

    print ("Saving all layers...")
    transfer_params = tf.get_collection("transfer_params")
    transfer_saver = tf.train.Saver(transfer_params)
    transfer_saver.save(sess, model_folder + '/transfer_model/all/' + '{}_transfer_params'.format(args.gym_env.replace('-', '_')))
    print ("All layers saved")

    print ("Saving without fc2 layer...")
    # Remove fc2 weights
    for param in transfer_params[:]:
        print (colored("\t{}".format(param.op.name), "green"))
        if param.op.name == "net_-1/fc2_weights" or param.op.name == "net_-1/fc2_biases":
            transfer_params.remove(param)
            print (colored("\t{} removed".format(param.op.name), "red"))

    transfer_saver = tf.train.Saver(transfer_params)
    transfer_saver.save(sess, model_folder + '/transfer_model/nofc2/' + '{}_transfer_params'.format(args.gym_env.replace('-', '_')))
    print ("Without fc2 layer saved")

    print ("Saving without fc1 layer...")
    # Remove fc1 weights
    for param in transfer_params[:]:
        print (colored("\t{}".format(param.op.name), "green"))
        if param.op.name == "net_-1/fc1_weights" or param.op.name == "net_-1/fc1_biases":
            transfer_params.remove(param)
            print (colored("\t{} removed".format(param.op.name), "red"))

    transfer_saver = tf.train.Saver(transfer_params)
    transfer_saver.save(sess, model_folder + '/transfer_model/nofc1/' + '{}_transfer_params'.format(args.gym_env.replace('-', '_')))
    print ("Without fc1 layer saved")

    # Remove conv3 weights
    if args.use_mnih_2015:
        print ("Saving without conv3 layer...")
        for param in transfer_params[:]:
            print (colored("\t{}".format(param.op.name), "green"))
            if param.op.name == "net_-1/conv3_weights" or param.op.name == "net_-1/conv3_biases":
                transfer_params.remove(param)
                print (colored("\t{} removed".format(param.op.name), "red"))

        transfer_saver = tf.train.Saver(transfer_params)
        transfer_saver.save(sess, model_folder + '/transfer_model/noconv3/' + '{}_transfer_params'.format(args.gym_env.replace('-', '_')))
        print ("Without conv3 layer saved")

    print ("Saving without conv2 layer...")
    # Remove conv2 weights
    for param in transfer_params[:]:
        print (colored("\t{}".format(param.op.name), "green"))
        if param.op.name == "net_-1/conv2_weights" or param.op.name == "net_-1/conv2_biases":
            transfer_params.remove(param)
            print (colored("\t{} removed".format(param.op.name), "red"))

    transfer_saver = tf.train.Saver(transfer_params)
    transfer_saver.save(sess, model_folder + '/transfer_model/noconv2/' + '{}_transfer_params'.format(args.gym_env.replace('-', '_')))
    print ("Without conv2 layer saved")

    print (colored('Data saved!', 'green'))
