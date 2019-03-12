#!/usr/bin/env python3
import argparse
import coloredlogs
import logging
import os
import pathlib

from collect_demo import CollectDemonstration
from common.game_state import GameState
from common.util import LogFormatter
from common.util import prepare_dir
from datetime import datetime

logger = logging.getLogger('get_demo')


def get_demo(args):
    if args.demo_memory_folder is not None:
        demo_memory_folder = 'collected_demo/{}'.format(args.demo_memory_folder)
    else:
        demo_memory_folder = 'collected_demo/{}'.format(args.gym_env.replace('-', '_'))

    if args.append_experiment_num is not None:
        demo_memory_folder += '_' + args.append_experiment_num

    demo_memory_folder = pathlib.Path(demo_memory_folder)

    if args.hostname is None:
        hostname = os.uname()[1]
    else:
        hostname = args.hostname

    prepare_dir(demo_memory_folder / 'log' / hostname, empty=False)
    prepare_dir(demo_memory_folder / 'data' / hostname, empty=False)

    episode_life = not args.not_episodic_life
    datetime_collected = datetime.today().strftime('%Y%m%d_%H%M%S')
    log_file = '{}.log'.format(datetime_collected)
    log_path = demo_memory_folder / 'log' / hostname
    fh = logging.FileHandler(str(log_path / log_file), mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = LogFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logging.getLogger('collect_demo').addHandler(fh)
    logging.getLogger('game_state').addHandler(fh)
    logging.getLogger('replay_memory').addHandler(fh)
    logging.getLogger('atari_wrapper').addHandler(fh)

    game_state = GameState(env_id=args.gym_env, display=True, human_demo=True,
                           episode_life=episode_life)
    collect_demo = CollectDemonstration(
        game_state,
        84, 84, 4,
        args.gym_env,
        folder=demo_memory_folder,
        create_movie=args.create_movie,
        hertz=args.hz,
        skip=args.skip)
    collect_demo.run_episodes(
        args.num_episodes,
        minutes_limit=args.demo_time_limit,
        demo_type=0,
        log_file=log_file,
        hostname=hostname)
    game_state.close()


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

    args = parser.parse_args()

    logger.info('Collecting demonstration...')
    get_demo(args)


if __name__ == "__main__":
    main()
