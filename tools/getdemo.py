#!/usr/bin/env python3
import logging
import os
import pathlib

from collect_demo import CollectDemonstration
from common.game_state import GameState
from common.util import LogFormatter
from common.util import prepare_dir
from datetime import datetime

logger = logging.getLogger('get_demo')


def get_demo(args, game=None, pause_onstart=True):
    gym_env = args.gym_env
    if game is not None:
        gym_env = game + 'NoFrameskip-v4'

    if args.demo_memory_folder is not None:
        demo_memory_folder = 'collected_demo/{}'.format(args.demo_memory_folder)
    else:
        demo_memory_folder = 'collected_demo/{}'.format(gym_env.replace('-', '_'))

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

    game_state = GameState(env_id=gym_env, display=True, human_demo=True,
                           episode_life=episode_life)
    collect_demo = CollectDemonstration(
        game_state,
        84, 84, 4,
        gym_env,
        folder=demo_memory_folder,
        create_movie=args.create_movie,
        hertz=args.hz,
        skip=args.skip,
        pause_onstart=pause_onstart)
    collect_demo.run_episodes(
        args.num_episodes,
        minutes_limit=args.demo_time_limit,
        demo_type=0,
        log_file=log_file,
        hostname=hostname)
    game_state.close()
