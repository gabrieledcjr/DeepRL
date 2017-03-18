#!/usr/bin/env python3
import argparse
import multiprocessing
import os
import numpy as np
import envs.gym_fun as game
from time import sleep
from termcolor import colored
from numpy.random import RandomState
from data_set import DataSet

def classify_demo(args):
    """
    python3 run_experiment.py pong --cuda-devices=0 --gpu-fraction=0.222 --optimizer=Adam --lr=0.0001 --decay=0.0 --momentum=0.0 --epsilon=0.001 --train-max-steps=150000 --batch=32 --eval-freq=500 --classify-demo
    """
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    import tensorflow as tf
    #from dqn_net_bn_class import DqnNetClass
    from dqn_net_class import DqnNetClass
    from classify_demo import ClassifyDemo

    if args.path is not None:
        path = args.path
    else:
        path = os.getcwd() + '/'

    if args.folder is not None:
        folder = '{}_{}'.format(args.env, args.folder)
    else:
        folder = '{}_networks_classifier_{}'.format(args.env, args.optimizer.lower())

    if args.cpu_only:
        device = '/cpu:0'
        gpu_options = None
    else:
        device = '/gpu:'+os.environ["CUDA_VISIBLE_DEVICES"]
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=True,
        intra_op_parallelism_threads=multiprocessing.cpu_count(),
        inter_op_parallelism_threads=multiprocessing.cpu_count()
    )

    with tf.device(device):
        game_state = game.GameState(game=args.env)
        if False: # Deterministic
            rng = np.random.RandomState(123456)
        else:
            rng = np.random.RandomState()
        D = DataSet(
            args.resized_height, args.resized_width,
            rng, args.replay_memory,
            args.phi_len, game_state.n_actions)
        net = DqnNetClass(
            args.resized_height, args.resized_width, args.phi_len,
            game_state.n_actions, args.env,
            optimizer=args.optimizer, learning_rate=args.lr,
            epsilon=args.epsilon, decay=args.decay, momentum=args.momentum,
            verbose=args.verbose, path=path, folder=folder)
        sess = tf.Session(config=config, graph=net.graph)
        net.initializer(sess)
        cd = ClassifyDemo(
            net, D, args.env, args.train_max_steps, args.batch, args.eval_freq,
            '{}_human_samples'.format(args.env))
        cd.run()

def collect_human_demo(args):
    """
    python3 run_experiment.py pong --demo-time-limit=5 --human-demo --file-num=1
    """
    from collect_human_demo import CollectHumanDemo

    if args.folder is not None:
        folder = '{}_{}'.format(args.env, args.folder)
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

def run_dqn(args):
    """
    Baseline:
    python3 run_experiment.py pong --cuda-devices=0 --optimizer=Adam --lr=0.0001 --decay=0.0 --momentum=0.0 --epsilon=0.001 --gpu-fraction=0.222
    python3 run_experiment.py pong --cuda-devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.01 --gpu-fraction=0.222

    Transfer with Human Memory:
    python3 run_experiment.py pong --cuda-devices=0 --optimizer=Adam --lr=0.0001 --decay=0.0 --momentum=0.0 --epsilon=0.001 --observe=0 --use-transfer --load-memory
    python3 run_experiment.py pong --cuda-devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.01 --observe=0 --use-transfer --load-memory
    python3 run_experiment.py breakout --cuda-devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.01 --observe=0 --use-transfer --load-memory --train-max-steps=20125000

    Transfer with Human Advice and Human Memory:
    python3 run_experiment.py pong --cuda-devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.01 --observe=0 --use-transfer --load-memory --use-human-model-as-advice --advice-confidence=0.75 --psi=0.9999979

    Human Advice only with Human Memory:
    python3 run_experiment.py pong --cuda-devices=0 --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.01 --observe=0 --load-memory --use-human-model-as-advice --advice-confidence=0.75 --psi=0.9999979
    """
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    import tensorflow as tf
    from experiment import Experiment
    from dqn_net import DqnNet
    from dqn_net_class import DqnNetClass

    if args.path is not None:
        path = args.path
    else:
        path = os.getcwd() + '/'

    if args.folder is not None:
        folder = '{}_{}'.format(args.env, args.folder)
    else:
        folder = '{}_networks_{}'.format(args.env, args.optimizer.lower())
        if args.use_transfer:
            folder = '{}_networks_transfer_{}'.format(args.env, args.optimizer.lower())
        if args.use_human_model_as_advice:
            folder = '{}_networks_transfer_w_advice_{}'.format(args.env, args.optimizer.lower())

    if args.cpu_only:
        device = '/cpu:0'
        gpu_options = None
    else:
        device = '/gpu:'+os.environ["CUDA_VISIBLE_DEVICES"]
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=True,
        intra_op_parallelism_threads=multiprocessing.cpu_count(),
        inter_op_parallelism_threads=multiprocessing.cpu_count()
    )

    game_state = game.GameState(game=args.env)
    human_net = None
    sess_human = None
    if args.use_human_model_as_advice:
        human_net = DqnNetClass(
            args.resized_height, args.resized_width,
            args.phi_len, game_state.n_actions, args.env,
            optimizer="Adam", learning_rate=0.0001, epsilon=0.001,
            decay=0., momentum=0., path=path,
            folder="{}_networks_classifier_{}".format(args.env, "adam"))
        sess_human = tf.Session(config=config, graph=human_net.graph)
        human_net.initializer(sess_human)
        human_net.load()

    with tf.Session(config=config) as sess:
        with tf.device(device):
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
                train_max_steps=args.train_max_steps,
                human_net=human_net, confidence=args.advice_confidence, psi=args.psi)
            experiment.run()
            sess_human.close()

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
    parser.add_argument('--final-epsilon', type=float, default=0.1)
    parser.add_argument('--init-epsilon', type=float, default=1.0)
    parser.add_argument('--replay-memory', type=int, default=1000000)
    parser.add_argument('--resized-width', type=int, default=84)
    parser.add_argument('--resized-height', type=int, default=84)
    parser.add_argument('--phi-len', type=int, default=4)
    parser.add_argument('--batch', type=int, default=32)

    parser.add_argument('--update-freq', type=int, default=4)
    parser.add_argument('--save-freq', type=int, default=125000)
    parser.add_argument('--eval-freq', type=int, default=250000)
    parser.add_argument('--eval-max-steps', type=int, default=125000)
    parser.add_argument('--train-max-steps', type=int, default=30125000)

    parser.add_argument('--c-freq', type=int, default=10000)
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

    parser.add_argument('--cuda-devices', type=str, default='0')
    parser.add_argument('--gpu-fraction', type=float, default=0.333)
    parser.add_argument('--cpu-only', action='store_true')
    parser.set_defaults(cpu_only=False)

    parser.add_argument('--use-transfer', action='store_true')
    parser.set_defaults(use_transfer=False)
    parser.add_argument('--transfer-folder', type=str, default=None)

    parser.add_argument('--use-human-model-as-advice', action='store_true')
    parser.set_defaults(use_human_model_as_advice=False)
    parser.add_argument('--advice-confidence', type=float, default=0.)
    parser.add_argument('--psi', type=float, default=0.)

    parser.add_argument('--load-memory', action='store_true')
    parser.set_defaults(load_memory=False)

    parser.add_argument('--human-demo', action='store_true')
    parser.set_defaults(human_demo=False)
    parser.add_argument('--random-action', action='store_true')
    parser.set_defaults(random_action=False)
    parser.add_argument('-n', '--file-num', type=int, default=1)
    parser.add_argument('--demo-time-limit', type=int, default=5) # 5 minutes

    parser.add_argument('--classify-demo', action='store_true')
    parser.set_defaults(classify_demo=False)

    args = parser.parse_args()

    if args.human_demo:
        print (colored('Collecting human demonstration...', 'green'))
        sleep(2)
        collect_human_demo(args)
    elif args.classify_demo:
        print (colored('Classifying human demonstration...', 'green'))
        sleep(2)
        classify_demo(args)
    else:
        print (colored('Running DQN...', 'green'))
        sleep(2)
        run_dqn(args)


if __name__ == "__main__":
    main()
