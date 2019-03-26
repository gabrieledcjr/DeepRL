#!/usr/bin/env python3
import argparse
import coloredlogs
import logging

from getdemo import get_demo
from multigame import multi_game

logger = logging.getLogger('get_demo')


def main():
    """Run main program.

    Requirements:
    sudo apt-get install python3-tk

    Usage:
    python3 get_demo.py --gym-env=PongNoFrameskip-v4 --num-episodes=5
        --demo-time-limit=20 --hz=60.0 --create-movie
    """
    coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()

    parser.add_argument('--gym-env', type=str, default='PongNoFrameskip-v4', help='OpenAi Gym environment ID')
    parser.add_argument('--not-episodic-life', action='store_true')
    parser.set_defaults(not_episodic_life=False)

    parser.add_argument('--skip', type=int, default=4, help='games is played with no skip but frames are saved every skip steps')

    parser.add_argument('--append-experiment-num', type=str, default=None)
    parser.add_argument('--num-episodes', type=int, default=5, help='number of episodes')
    parser.add_argument('--demo-time-limit', type=int, default=5, help='time limit per episode')
    parser.add_argument('--create-movie', action='store_true')
    parser.set_defaults(create_movie=False)
    parser.add_argument('--demo-memory-folder', type=str, default=None)
    parser.add_argument('--hz', type=float, default=60.0, help='game update frequency')

    parser.add_argument('--hostname', type=str, default=None)

    parser.add_argument('--multi-game', action='store_true')
    parser.set_defaults(multi_game=False)

    args = parser.parse_args()

    logger.info('Collecting demonstration...')

    if args.multi_game:
        multi_game(args)
    else:
        get_demo(args)


if __name__ == "__main__":
    main()
