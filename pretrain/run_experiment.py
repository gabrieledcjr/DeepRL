#!/usr/bin/env python3
"""Run experiments for pre-training with human demonstration.

Examples:
    Multi-class classification
        $ python3 pretrain/run_experiment.py
            --gym-env=PongNoFrameskip-v4
            --classify-demo --use-mnih-2015
            --train-max-steps=150000 --batch-size=32

    MTL one class vs. all classes
        $ python3 pretrain/run_experiment.py
            --gym-env=PongNoFrameskip-v4
            --classify-demo --onevsall-mtl --use-mnih-2015
            --train-max-steps=150000 --batch-size=32

    Autoencoder
        $ python3 pretrain/run_experiment.py
            --gym-env=PongNoFrameskip-v4
            --ae-classify-demo --use-mnih-2015
            --train-max-steps=150000 --batch-size=32
"""
import argparse
import coloredlogs
import logging

from ae_classify_demo import ae_classify_demo
from classify_demo import classify_demo
from extract_transfer_layers import extract_layers
from time import sleep

logger = logging.getLogger()


def main():
    """Contain all arguments for command line execution of experiment."""
    coloredlogs.install(
        level='DEBUG',
        fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()

    parser.add_argument('--gym-env', type=str, default='PongNoFrameskip-v4',
                        help='OpenAi Gym environment ID')

    parser.add_argument('--classify-demo', action='store_true')
    parser.set_defaults(classify_demo=False)

    parser.add_argument('--ae-classify-demo', action='store_true',
                        help='Use Autoencoder then classify')
    parser.set_defaults(ae_classify_demo=False)

    parser.add_argument('--sae-classify-demo', action='store_true',
                        help='Use Supervised Autoencoder')
    parser.set_defaults(sae_classify_demo=False)
    parser.add_argument(
        '--loss-function', type=str, default='mse',
        help='mse (mean squared error) or bce (binary cross entropy)')
    parser.add_argument('--sl-loss-weight', type=float, default=1.0,
                        help='weighted classification loss')

    parser.add_argument('--use-denoising', action='store_true')
    parser.set_defaults(use_denoising=False)
    parser.add_argument('--noise-factor', type=float, default=0.3)

    parser.add_argument('--tied-weights', action='store_true',
                        help='Autoencoder with tied weights')
    parser.set_defaults(tied_weights=False)

    parser.add_argument('--input-shape', type=int,
                        default=84, help='84x84 as default')
    parser.add_argument('--padding', type=str, default='VALID',
                        help='VALID or SAME')

    parser.add_argument('--model-folder', type=str, default=None)

    parser.add_argument('--train-max-steps', type=float, default=10 * 10 ** 7,
                        help='maximum training steps')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='batch size')

    parser.add_argument('--optimizer', type=str, default='rms',
                        help='rms or adam')
    parser.add_argument('--learn-rate', type=float, default=7e-4,
                        help='learning rate for RMS/Adam optimizer')
    parser.add_argument('--opt-alpha', type=float, default=0.99,
                        help='decay parameter for RMS optimizer')
    parser.add_argument('--opt-epsilon', type=float, default=1e-5,
                        help='epsilon parameter for RMS/Adam optimizer')

    parser.add_argument('--l1-beta', type=float, default=0.,
                        help='L1 regularization beta')
    parser.add_argument('--l2-beta', type=float, default=1e-4,
                        help='L2 regularization beta')
    parser.add_argument('--grad-norm-clip', type=float, default=None,
                        help='gradient norm clipping')

    parser.add_argument('--eval-freq', type=int, default=5000,
                        help='evaluation frequency')

    parser.add_argument('--cpu-only', action='store_true', help='use CPU only')
    parser.set_defaults(cpu_only=False)
    parser.add_argument('--gpu-fraction', type=float, default=0.4)
    parser.add_argument('--cuda-devices', type=str, default='')

    parser.add_argument(
        '--use-mnih-2015', action='store_true',
        help='use Mnih et al [2015] network architecture (3 conv layers)')
    parser.set_defaults(use_mnih_2015=False)

    parser.add_argument('--demo-memory-folder', type=str, default=None)
    parser.add_argument('--append-experiment-num', type=str, default=None)

    parser.add_argument('--demo-ids', type=str, default=None,
                        help='demo ids separated by comma')

    parser.add_argument(
        '--exclude-num-demo-ep', type=int, default=0,
        help='exclude number of demo episodes from classification training')

    parser.add_argument('--use-batch-proportion', action='store_true')
    parser.set_defaults(use_batch_proportion=False)

    parser.add_argument('--onevsall-mtl', action='store_true')
    parser.set_defaults(onevsall_mtl=False)
    parser.add_argument('--exclude-noop', action='store_true')
    parser.set_defaults(exclude_noop=False)

    parser.add_argument('--extract-transfer-layers', action='store_true')
    parser.set_defaults(extract_transfer_layers=False)

    parser.add_argument('--use-grad-cam', action='store_true')
    parser.set_defaults(use_grad_cam=False)

    parser.add_argument('--use-sil', action='store_true',
                        help='self imitation learning loss')
    parser.set_defaults(use_sil=False)

    args = parser.parse_args()

    if args.extract_transfer_layers:
        logger.info('Extracting transfer layers...')
        sleep(2)
        extract_layers(args)
    elif args.classify_demo:
        logger.info('Classifying human demonstration...')
        sleep(2)
        classify_demo(args)
    elif args.ae_classify_demo or args.sae_classify_demo:
        logger.info('Use autoencoder and classify human demonstration...')
        sleep(2)
        ae_classify_demo(args)


if __name__ == "__main__":
    main()
