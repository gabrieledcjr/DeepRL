# async_deep_reinforce

Asynchronous deep reinforcement learning

## About

An attempt to repdroduce Google Deep Mind's paper "Asynchronous Methods for Deep Reinforcement Learning."

http://arxiv.org/abs/1602.01783

Asynchronous Advantage Actor-Critic (A3C) method for playing "Atari Pong" is implemented with TensorFlow.
Both A3C-FF and A3C-LSTM are implemented.

Learning result movment after 26 hours (A3C-FF) is like this.

[![Learning result after 26 hour](http://narr.jp/private/miyoshi/deep_learning/a3c_preview_image.jpg)](https://youtu.be/ZU71YdAedZs)

Any advice or suggestion is strongly welcomed in issues thread.

https://github.com/miyosuda/async_deep_reinforce/issues/1

## Install Open AI Gym [Atari]

First you need to install Atari gym environment. More documentation [here](https://github.com/openai/gym).

    $ pip install gym[atari]

## How to run

To train,

    $python a3c.py --gym-env=PongDeterministic-v3 \
        --parallel-size=8 --initial-learn-rate=7e-4 \
	--use-lstm --use-mnih-2015

To display the result with game play (IN PROGRESS),

    $python a3c_disp.py --gym-env=PongDeterministic-v3 \
        --parallel-size=8 --initial-learn-rate=7e-4 \
	--use-lstm --use-mnih-2015

## Using GPU
To enable gpu, include the "--use-gpu" argument.

When running with 8 parallel game environemts, speeds of GPU (GTX980Ti) and CPU(Core i7 6700) were like this.

|type | A3C-FF            |A3C-LSTM          |
|-----|-------------------|------------------|
| GPU | 821 steps per sec |416 steps per sec |
| CPU | 472 steps per sec |243 steps per sec |


## Result
Score plots of local threads of pong in 24h were like these. (with GTX980Ti)

### A3C-FF 
(70.9 million global steps in 24 hours)

![A3C-FF scores](https://github.com/miyosuda/async_deep_reinforce/blob/master/docs/graph_24h.png)

### A3C-LSTM 
(35.9 million global steps in 24 hours)

![A3C-LSTM scores](https://github.com/miyosuda/async_deep_reinforce/blob/master/docs/graph_24h_lstm.png)

Scores are not averaged using global network unlike the original paper.

## References

This project uses setting written in muupan's wiki [muuupan/async-rl] (https://github.com/muupan/async-rl/wiki)


## Acknowledgements

- [@aravindsrinivas](https://github.com/aravindsrinivas) for providing information for some of the hyper parameters.

