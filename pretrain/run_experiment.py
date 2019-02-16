#!/usr/bin/env python3
import argparse
import coloredlogs, logging

from time import sleep
from classify_demo import classify_demo
from extract_transfer_layers import extract_layers

logger = logging.getLogger()

def main():
    coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()

    parser.add_argument('--gym-env', type=str, default='PongNoFrameskip-v4', help='OpenAi Gym environment ID')

    parser.add_argument('--classify-demo', action='store_true')
    parser.set_defaults(classify_demo=False)

    parser.add_argument('--model-folder', type=str, default=None)

    parser.add_argument('--train-max-steps', type=float, default=10 * 10 ** 7, help='maximum training steps')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')

    parser.add_argument('--learn-rate', type=float, default=7e-4, help='learning rate for optimizer')
    parser.add_argument('--opt-alpha', type=float, default=0.99, help='decay parameter for optimizer')
    parser.add_argument('--opt-epsilon', type=float, default=1e-5, help='epsilon parameter for optimizer')

    parser.add_argument('--l1-beta', type=float, default=0., help='L1 regularization beta')
    parser.add_argument('--l2-beta', type=float, default=1e-4, help='L2 regularization beta')
    parser.add_argument('--grad-norm-clip', type=float, default=0.5, help='gradient norm clipping')

    parser.add_argument('--eval_freq', type=int, default=5000, help='evaluation frequency')

    parser.add_argument('--cpu-only', action='store_true', help='use CPU only')
    parser.set_defaults(cpu_only=False)
    parser.add_argument('--gpu-fraction', type=float, default=0.4)
    parser.add_argument('--cuda-devices', type=str, default='')

    parser.add_argument('--use-mnih-2015', action='store_true', help='use Mnih et al [2015] network architecture (3 conv layers)')
    parser.set_defaults(use_mnih_2015=False)

    parser.add_argument('--demo-memory-folder', type=str, default=None)
    parser.add_argument('--append-experiment-num', type=str, default=None)

    parser.add_argument('--demo-ids', type=str, default=None, help='demo ids separated by comma')

    parser.add_argument('--exclude-num-demo-ep', type=int, default=0, help='exclude number of demo episodes from classification training')
    parser.add_argument('--exclude-k-steps-bad-state', type=int, default=0, help='exclude k number of steps from a bad state (negative reward or life loss)')
    parser.add_argument('--weighted-cross-entropy', action='store_true')
    parser.set_defaults(weighted_cross_entropy=False)

    parser.add_argument('--use-batch-proportion', action='store_true')
    parser.set_defaults(use_batch_proportion=False)

    parser.add_argument('--onevsall-mtl', action='store_true')
    parser.set_defaults(onevsall_mtl=False)
    parser.add_argument('--exclude-noop', action='store_true')
    parser.set_defaults(exclude_noop=False)

    parser.add_argument('--extract-transfer-layers', action='store_true')
    parser.set_defaults(extract_transfer_layers=False)

    args = parser.parse_args()

    if args.extract_transfer_layers:
        logger.info('Extracting transfer layers...')
        sleep(2)
        extract_layers(args)
    elif args.classify_demo:
        logger.info('Classifying human demonstration...')
        sleep(2)
        classify_demo(args)


if __name__ == "__main__":
    main()
