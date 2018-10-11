# async_deep_reinforce

Asynchronous deep reinforcement learning

## About

An attempt to repdroduce Google Deep Mind's paper "Asynchronous Methods for Deep Reinforcement Learning."

http://arxiv.org/abs/1602.01783

Asynchronous Advantage Actor-Critic (A3C) method for playing "Atari Pong" is implemented with TensorFlow.
Both A3C-FF and A3C-LSTM are implemented.

## Install Open AI Gym [Atari]

First you need to install Atari gym environment. More documentation [here](https://github.com/openai/gym).

    $ pip install gym[atari]

## How to run

To train,

    $python a3c.py --gym-env=PongNoFrameskip-v4 \
        --parallel-size=8 --initial-learn-rate=7e-4 \
        --use-lstm --use-mnih-2015

To display the result with game play (IN PROGRESS),

    $python a3c_disp.py --gym-env=PongNoFrameskip-v4 \
        --parallel-size=8 --initial-learn-rate=7e-4 \
        --use-lstm --use-mnih-2015

## Acknowledgements

- [@aravindsrinivas](https://github.com/aravindsrinivas) for providing information for some of the hyper parameters.
