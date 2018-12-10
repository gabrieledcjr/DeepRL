#!/usr/bin/env python3
import os
import argparse
import coloredlogs, logging

from time import sleep
from a3c import run_a3c

logger = logging.getLogger()

def main():
    coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()

    parser.add_argument('--parallel-size', type=int, default=16, help='parallel thread size')
    parser.add_argument('--gym-env', type=str, default='PongNoFrameskip-v4', help='OpenAi Gym environment ID')

    parser.add_argument('--local-t-max', type=int, default=20, help='repeat step size')
    parser.add_argument('--rmsp-alpha', type=float, default=0.99, help='decay parameter for RMSProp')
    parser.add_argument('--rmsp-epsilon', type=float, default=1e-5, help='epsilon parameter for RMSProp')

    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--append-experiment-num', type=str, default=None)

    parser.add_argument('--initial-learn-rate', type=float, default=7e-4, help='initial learning rate for RMSProp')
    parser.add_argument('--initial-alpha-low', type=float, default=1e-4, help='log_uniform low limit for learning rate')
    parser.add_argument('--initial-alpha-high', type=float, default=1e-2, help='log_uniform high limit for learning rate')
    parser.add_argument('--initial-alpha-log-rate', type=float, default=0.4226, help='log_uniform interpolate rate for learning rate (around 7 * 10^-4)')

    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--entropy-beta', type=float, default=0.01, help='entropy regularization constant')
    parser.add_argument('--max-time-step', type=float, default=10 * 10**7, help='maximum time step, also use to anneal learning rate')
    parser.add_argument('--max-time-step-fraction', type=float, default=1., help='ovverides maximum time step by a fraction')
    parser.add_argument('--grad-norm-clip', type=float, default=0.5, help='gradient norm clipping')

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

    parser.add_argument('--log-interval', type=int, default=500)
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
    parser.add_argument('--finetune-upper-layers-only', action='store_true')
    parser.set_defaults(finetune_upper_layers_only=False)

    parser.add_argument('--load-memory', action='store_true')
    parser.set_defaults(load_memory=False)
    parser.add_argument('--demo-memory-folder', type=str, default=None)
    parser.add_argument('--train-with-demo-num-steps', type=int, default=0, help='pretraining number of steps/frames')
    parser.add_argument('--train-with-demo-num-epochs', type=int, default=0, help='pretraining number of epochs')
    parser.add_argument('--demo-t-max', type=int, default=20, help='demo repeat step size')
    parser.add_argument('--demo-entropy-beta', type=float, default=0.01, help='demo entropy regularization constant')

    parser.add_argument('--use-demo-threads', action='store_true')
    parser.set_defaults(use_demo_threads=False)
    parser.add_argument('--max-steps-threads-as-demo', type=int, default=1000000)

    parser.add_argument('--load-demo-cam', action='store_true')
    parser.set_defaults(load_demo_cam=False)
    parser.add_argument('--demo-cam-id', type=str, default=None, help='demo id for cam')

    parser.add_argument('--l2-beta', type=float, default=0., help='L2 regularization beta')
    parser.add_argument('--model-folder', type=str, default=None)

    parser.add_argument('--onevsall-mtl', action='store_true')
    parser.set_defaults(onevsall_mtl=False)

    # Alternatives to reward clipping
    parser.add_argument('--unclipped-reward', action='store_true', help='use raw reward')
    parser.set_defaults(unclipped_reward=False)
    # DQfD Hester, et. al 2017
    parser.add_argument('--log-scale-reward', action='store_true', help='use log scale reward r = sign(r) * log(1 + abs(r)) from DQfD (Hester et. al)')
    parser.set_defaults(log_scale_reward=False)
    # Ape-X Pohlen, et. al 2018
    parser.add_argument('--transformed-bellman', action='store_true', help='use transofrmed bellman equation')
    parser.set_defaults(transformed_bellman=False)

    parser.add_argument('--load-pretrained-model', action='store_true')
    parser.set_defaults(load_pretrained_model=False)
    parser.add_argument('--pretrained-model-folder', type=str, default=None)
    parser.add_argument('--use-pretrained-model-as-advice', action='store_true', help='use human model as advice (Wang et. al)')
    parser.set_defaults(use_pretrained_model_as_advice=False)
    parser.add_argument('--use-pretrained-model-as-reward-shaping', action='store_true', help='use human model for reward shaping (Brys et. al)')
    parser.set_defaults(use_pretrained_model_as_reward_shaping=False)

    args = parser.parse_args()

    logger.info('Running A3C...')
    sleep(2)
    run_a3c(args)


if __name__ == "__main__":
    main()
