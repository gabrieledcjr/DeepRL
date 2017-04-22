# -*- coding: utf-8 -*-
import os
import argparse

from time import sleep
from termcolor import colored
from a3c import run_a3c
from classify_demo import classify_demo
from collect_demo import get_demo

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--parallel-size', type=int, default=16, help='parallel thread size')
    parser.add_argument('--gym-env', type=str, default='PongDeterministic-v3', help='OpenAi Gym environment ID')

    parser.add_argument('--local-t-max', type=int, default=20, help='repeat step size')
    parser.add_argument('--rmsp-alpha', type=float, default=0.99, help='decay parameter for RMSProp')
    parser.add_argument('--rmsp-epsilon', type=float, default=0.1, help='epsilon parameter for RMSProp')

    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--log-file', type=str, default='tmp/a3c_log')

    parser.add_argument('--initial-learn-rate', type=float, default=0, help='initial learning rate for RMSProp')
    parser.add_argument('--initial-alpha-low', type=float, default=1e-4, help='log_uniform low limit for learning rate')
    parser.add_argument('--initial-alpha-high', type=float, default=1e-2, help='log_uniform high limit for learning rate')
    parser.add_argument('--initial-alpha-log-rate', type=float, default=0.4226, help='log_uniform interpolate rate for learning rate (around 7 * 10^-4)')

    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--entropy-beta', type=float, default=0.01, help='entropy regularization constant')
    parser.add_argument('--max-time-step', type=float, default=10 * 10**7, help='maximum time step')
    parser.add_argument('--grad-norm-clip', type=float, default=40.0, help='gradient norm clipping')

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

    parser.add_argument('--collect-demo', action='store_true')
    parser.set_defaults(collect_demo=False)
    parser.add_argument('--num-episodes', type=int, default=5, help='number of episodes')
    parser.add_argument('--demo-time-limit', type=int, default=5, help='time limit per episode')
    parser.add_argument('--demo-memory-folder', type=str, default=None)

    parser.add_argument('--classify-demo', action='store_true')
    parser.set_defaults(classify_demo=False)
    parser.add_argument('--model-folder', type=str, default=None)

    args = parser.parse_args()

    if args.collect_demo:
        print (colored('Collecting demonstration...', 'green'))
        sleep(2)
        get_demo(args)
    elif args.classify_demo:
        print (colored('Classifying human demonstration...', 'green'))
        sleep(2)
        classify_demo(args)
    else:
        print (colored('Running A3C...', 'green'))
        time.sleep(2)
        run_a3c(args)


if __name__ == "__main__":
    main()
