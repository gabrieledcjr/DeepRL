#!/usr/bin/env python
import argparse
import multiprocessing
import os
import numpy as np
import envs.gym_fun as game
from numpy.random import RandomState
from data_set import DataSet


def collect_human_demo(args):
    """
    python3 run_dqn.py pong --demo-time-limit=5 --human-demo --file-num=1
    """
    from collect_human_demo import CollectHumanDemo

    if args.folder is not None:
        folder = args.folder
    else:
        folder = '{}_human_samples'.format(args.env)

    game_state = game.GameState(human_demo=True, frame_skip=1, game=args.env)
    if False: # Deterministic
        rng = RandomState(123456)
    else:
        rng = RandomState()
    D = DataSet(
        args.resized_height, args.resized_width,
        rng, (args.demo_time_limit * 5000),
        args.phi_len, game_state.n_actions)

    collect_demo = CollectHumanDemo(
        game_state, args.resized_height, args.resized_width, args.phi_len,
        args.env, D, folder=folder, sample_num=args.file_num
    )
    collect_demo.run(minutes_limit=args.demo_time_limit, random_action=args.random_action)

def run_experiment(args):
    """
    Baseline:
    python3 run_dqn.py pong --cuda_devices=0 --optimizer=Adam --lr=0.0001 --decay=0.0 --momentum=0.0 --epsilon=0.001 --gpu_fraction=0.222
    python3 run_dqn.py pong --cuda_devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.01 --gpu_fraction=0.222
    python3 run_dqn.py pong --cuda_devices=0 --optimizer=Graves --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.01

    Transfer:
    python3 run_dqn.py pong --cuda_devices=0 --optimizer=Adam --lr=0.0001 --decay=0.0 --momentum=0.0 --epsilon=0.001 --observe=0 --use-transfer --load-memory
    python3 run_dqn.py pong --cuda_devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.01 --observe=0 --use-transfer --load-memory
    python3 run_dqn.py pong --cuda_devices=0 --optimizer=Graves --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.01 --observe=0 --use-transfer --load-memory

    python3 run_dqn.py breakout --cuda_devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.01 --observe=0 --use-transfer --load-memory --train_max_steps=20125000
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    import tensorflow as tf
    from experiment import Experiment
    from dqn_net import DqnNet

    if args.path is not None:
        path = args.path
    else:
        path = os.getcwd() + '/'

    if args.folder is not None:
        folder = args.folder
    else:
        if args.use_transfer:
            folder = '{}_networks_transfer_{}'.format(args.env, args.optimizer.lower())
        else:
            folder = '{}_networks_{}'.format(args.env, args.optimizer.lower())

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=True,
        intra_op_parallelism_threads=multiprocessing.cpu_count()
    )

    with tf.Session(config=config) as sess:
        with tf.device('/gpu:'+os.environ["CUDA_VISIBLE_DEVICES"]):
            game_state = game.GameState(game=args.env)
            if False: # Deterministic
                rng = RandomState(123456)
            else:
                rng = RandomState()
            D = DataSet(
                args.resized_height, args.resized_width,
                rng, args.replay_memory,
                args.phi_len, game_state.n_actions)

            # baseline learning
            if not args.use_transfer:
                net = DqnNet(
                    sess, args.resized_height, args.resized_width, args.phi_len,
                    game_state.n_actions, args.env, gamma=args.gamma, copy_interval=args.c_freq,
                    optimizer=args.optimizer, learning_rate=args.lr,
                    epsilon=args.epsilon, decay=args.decay, momentum=args.momentum,
                    verbose=args.verbose, path=path, folder=folder,
                    slow=args.use_slow, tau=args.tau)

            # transfer using existing model
            else:
                if args.transfer_folder is not None:
                    transfer_folder = args.transfer_folder
                else:
                    # Always load adam model
                    transfer_folder = "{}_networks_classifier_{}/transfer_model".format(args.env, "adam")

                net = DqnNet(
                    sess, args.resized_height, args.resized_width, args.phi_len,
                    game_state.n_actions, args.env, gamma=args.gamma, copy_interval=args.c_freq,
                    optimizer=args.optimizer, learning_rate=args.lr,
                    epsilon=args.epsilon, decay=args.decay, momentum=args.momentum,
                    verbose=args.verbose, path=path, folder=folder,
                    slow=args.use_slow, tau=args.tau,
                    transfer=True, transfer_folder=transfer_folder)

            experiment = Experiment(
                sess, net, game_state, args.resized_height, args.resized_width,
                args.phi_len, args.batch, args.env,
                args.gamma, args.observe, args.explore, args.final_epsilon,
                args.init_epsilon, D,
                args.update_freq, args.save_freq, args.eval_freq,
                args.eval_max_steps, args.c_freq,
                path, folder, load_human_memory=args.load_memory,
                train_max_steps=args.train_max_steps)
            experiment.run()

def main():

    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Prevent numpy from using multiple threads
    # os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--observe', type=int, default=50000)
    parser.add_argument('--explore', type=int, default=1000000)
    parser.add_argument('--final_epsilon', type=float, default=0.1)
    parser.add_argument('--init_epsilon', type=float, default=1.0)
    parser.add_argument('--replay_memory', type=int, default=1000000)
    parser.add_argument('--resized_width', type=int, default=84)
    parser.add_argument('--resized_height', type=int, default=84)
    parser.add_argument('--batch', type=int, default=32)

    parser.add_argument('--phi_len', type=int, default=4)
    parser.add_argument('--update_freq', type=int, default=4)

    parser.add_argument('--save_freq', type=int, default=125000)
    parser.add_argument('--eval_freq', type=int, default=250000)
    parser.add_argument('--eval_max_steps', type=int, default=125000)
    parser.add_argument('--train_max_steps', type=int, default=30125000)

    parser.add_argument('--c_freq', type=int, default=10000)
    parser.add_argument('--use-slow', action='store_true')
    parser.set_defaults(use_slow=False)
    parser.add_argument('--tau', type=float, default=1.)

    parser.add_argument('--optimizer', type=str, default='RMS')
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--decay', type=float, default=0.95)
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--epsilon', type=float, default=0.01)

    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument('--gpu_fraction', type=float, default=0.333)

    parser.add_argument('--use-transfer', action='store_true')
    parser.set_defaults(use_transfer=False)
    parser.add_argument('--transfer_folder', type=str, default=None)

    parser.add_argument('--cuda_devices', type=str, default='0')

    parser.add_argument('--load-memory', action='store_true')
    parser.set_defaults(load_memory=False)

    parser.add_argument('--human-demo', action='store_true')
    parser.set_defaults(human_demo=False)
    parser.add_argument('--random-action', action='store_true')
    parser.set_defaults(random_action=False)
    parser.add_argument('-n', '--file-num', type=int, default=1)
    parser.add_argument('--demo-time-limit', type=int, default=5) # 5 minutes

    args = parser.parse_args()

    if args.human_demo:
        collect_human_demo(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
