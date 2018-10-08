#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name='deep_rl',
    description='',
    url='https://github.com/gabrieledcjr/DeepRL',
    author='Gabriel de la Cruz',
    author_email='gabrieledcjr@gmail.com',
    license='',
    packages=[package for package in find_packages()
                if package.startswith('common')],
    install_requires=['gym', 'atari_py', 'coloredlogs',
        'termcolor', 'pyglet', 'tables', 'matplotlib',
        'numpy', 'opencv-python']
)
