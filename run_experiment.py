# -*- coding: utf-8 -*-
import os
import argparse

from time import sleep
from termcolor import colored
from a3c import run_a3c
from classify_demo import classify_demo
from collect_demo import get_demo
from extract_transfer_layers import extract_layers

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--parallel-size', type=int, default=16, help='parallel thread size')
    parser.add_argument('--gym-env', type=str, default='PongDeterministic-v3', help='OpenAi Gym environment ID')

    parser.add_argument('--local-t-max', type=int, default=20, help='repeat step size')
    parser.add_argument('--rmsp-alpha', type=float, default=0.99, help='decay parameter for RMSProp')
    parser.add_argument('--rmsp-epsilon', type=float, default=0.1, help='epsilon parameter for RMSProp')

    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--append-experiment-num', type=str, default=None)

    parser.add_argument('--initial-learn-rate', type=float, default=0, help='initial learning rate for RMSProp')
    parser.add_argument('--initial-alpha-low', type=float, default=1e-4, help='log_uniform low limit for learning rate')
    parser.add_argument('--initial-alpha-high', type=float, default=1e-2, help='log_uniform high limit for learning rate')
    parser.add_argument('--initial-alpha-log-rate', type=float, default=0.4226, help='log_uniform interpolate rate for learning rate (around 7 * 10^-4)')

    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--entropy-beta', type=float, default=0.01, help='entropy regularization constant')
    parser.add_argument('--max-time-step', type=float, default=10 * 10**7, help='maximum time step, also use to anneal learning rate')
    parser.add_argument('--max-time-step-fraction', type=float, default=1., help='ovverides maximum time step by a fraction')
    parser.add_argument('--grad-norm-clip', type=float, default=40.0, help='gradient norm clipping')

    parser.add_argument('--eval-freq', type=int, default=1000000)
    parser.add_argument('--eval-max-steps', type=int, default=125000)

    parser.add_argument('--use-gpu', action='store_true', help='use GPU')
    parser.set_defaults(use_gpu=False)
    parser.add_argument('--gpu-fraction', type=float, default=0.4)
    parser.add_argument('--cuda-devices', type=str, default='')

    parser.add_argument('--use-lstm', action='store_true', help='use LSTM')
    parser.set_defaults(use_lstm=False)

    parser.add_argument('--use-mnih-2015', action='store_true', help='use Mnih et al [2015] network architecture (3 conv layers)')
    parser.set_defaults(use_mnih_2015=False)

    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--performance-log-interval', type=int, default=1000)

    parser.add_argument('--use-transfer', action='store_true')
    parser.set_defaults(use_transfer=False)
    parser.add_argument('--transfer-folder', type=str, default=None)
    parser.add_argument('--not-transfer-fc2', action='store_true')
    parser.set_defaults(not_transfer_fc2=False)
    parser.add_argument('--not-transfer-fc1', action='store_true')
    parser.set_defaults(not_transfer_fc1=False)
    parser.add_argument('--not-transfer-conv3', action='store_true')
    parser.set_defaults(not_transfer_conv3=False)
    parser.add_argument('--not-transfer-conv2', action='store_true')
    parser.set_defaults(not_transfer_conv2=False)

    parser.add_argument('--collect-demo', action='store_true')
    parser.set_defaults(collect_demo=False)
    parser.add_argument('--num-episodes', type=int, default=5, help='number of episodes')
    parser.add_argument('--demo-time-limit', type=int, default=5, help='time limit per episode')

    parser.add_argument('--load-memory', action='store_true')
    parser.set_defaults(load_memory=False)
    parser.add_argument('--demo-memory-folder', type=str, default=None)
    parser.add_argument('--train-with-demo-num-steps', type=int, default=0, help='pretraining number of steps/frames')
    parser.add_argument('--train-with-demo-num-epochs', type=int, default=0, help='pretraining number of epochs')

    parser.add_argument('--use-demo-threads', action='store_true')
    parser.set_defaults(use_demo_threads=False)
    parser.add_argument('--max-steps-threads-as-demo', type=int, default=1000000)

    parser.add_argument('--classify-demo', action='store_true')
    parser.set_defaults(classify_demo=False)
    parser.add_argument('--l1-beta', type=float, default=0., help='L1 regularization beta')
    parser.add_argument('--l2-beta', type=float, default=0., help='L2 regularization beta')
    parser.add_argument('--model-folder', type=str, default=None)

    parser.add_argument('--extract-transfer-layers', action='store_true')
    parser.set_defaults(extract_transfer_layers=False)

    parser.add_argument('--adaptive-reward', action='store_true')
    parser.set_defaults(adaptive_reward=False)

    parser.add_argument('--auto-start', action='store_true')
    parser.set_defaults(auto_start=False)

    parser.add_argument('--not-gae', action='store_true', help='compute gradients for V and pi separately')
    parser.set_defaults(not_gae=False)

    args = parser.parse_args()

    if args.extract_transfer_layers:
        print (colored('Extracting transfer layers...', 'green'))
        sleep(2)
        extract_layers(args)
    elif args.collect_demo:
        print (colored('Collecting demonstration...', 'green'))
        sleep(2)
        get_demo(args)
    elif args.classify_demo:
        print (colored('Classifying human demonstration...', 'green'))
        sleep(2)
        classify_demo(args)
    else:
        print (colored('Running A3C...', 'green'))
        sleep(2)
        run_a3c(args)


if __name__ == "__main__":
    main()
