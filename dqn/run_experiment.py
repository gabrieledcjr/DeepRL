#!/usr/bin/env python3
import argparse
import coloredlogs, logging

from time import sleep
from dqn import run_dqn

logger = logging.getLogger()
coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s')


def main():
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()

    parser.add_argument('--gym-env', type=str, default='PongNoFrameskip-v4', help='OpenAi Gym environment ID')
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
    parser.add_argument('--save-freq', type=int, default=1000000)
    parser.add_argument('--eval-freq', type=int, default=1000000)
    parser.add_argument('--eval-max-steps', type=int, default=125000)
    parser.add_argument('--train-max-steps', type=int, default=41 * 10**6)
    parser.add_argument('--grad-norm-clip', type=float, default=None, help='gradient norm clipping')

    parser.add_argument('--c-freq', type=int, default=10000)
    parser.add_argument('--use-slow', action='store_true')
    parser.set_defaults(use_slow=False)
    parser.add_argument('--tau', type=float, default=1.)

    parser.add_argument('--optimizer', type=str, default='RMS')
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--decay', type=float, default=0.95)
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--epsilon', type=float, default=0.00001)

    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--append-experiment-num', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.set_defaults(verbose=False)

    parser.add_argument('--cuda-devices', type=str, default='')
    parser.add_argument('--gpu-fraction', type=float, default=0.333)
    parser.add_argument('--cpu-only', action='store_true')
    parser.set_defaults(cpu_only=False)

    parser.add_argument('--use-transfer', action='store_true')
    parser.set_defaults(use_transfer=False)
    parser.add_argument('--transfer-folder', type=str, default=None)
    parser.add_argument('--not-transfer-conv2', action='store_true')
    parser.set_defaults(not_transfer_conv2=False)
    parser.add_argument('--not-transfer-conv3', action='store_true')
    parser.set_defaults(not_transfer_conv3=False)
    parser.add_argument('--not-transfer-fc1', action='store_true')
    parser.set_defaults(not_transfer_fc1=False)
    parser.add_argument('--not-transfer-fc2', action='store_true')
    parser.set_defaults(not_transfer_fc2=False)

    parser.add_argument('--use-human-model-as-advice', action='store_true')
    parser.set_defaults(use_human_model_as_advice=False)
    parser.add_argument('--advice-confidence', type=float, default=0.)
    parser.add_argument('--advice-folder', type=str, default=None)
    parser.add_argument('--psi', type=float, default=0.)

    parser.add_argument('--load-memory', action='store_true')
    parser.set_defaults(load_memory=False)
    parser.add_argument('--demo-memory-folder', type=str, default=None)

    parser.add_argument('--train-with-demo-steps', type=int, default=0)

    # Alternatives to reward clipping
    parser.add_argument('--unclipped-reward', action='store_true', help='use raw reward')
    parser.set_defaults(unclipped_reward=False)
    # DQfD Hester, et. al 2017
    parser.add_argument('--log-scale-reward', action='store_true', help='use log scale reward r = sign(r) * log(1 + abs(r)) from DQfD (Hester et. al)')
    parser.set_defaults(log_scale_reward=False)
    # Ape-X Pohlen, et. al 2018
    parser.add_argument('--transformed-bellman', action='store_true', help='use transofrmed bellman equation')
    parser.set_defaults(transformed_bellman=False)
    parser.add_argument('--target-consistency', action='store_true', help='use target consistency (TC) loss')
    parser.set_defaults(target_consistency=False)

    args = parser.parse_args()

    logger.info('Running DQN...')
    sleep(2)
    run_dqn(args)


if __name__ == "__main__":
    main()
