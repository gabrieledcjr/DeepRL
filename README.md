# Pre-training with Human Demonstration in Deep Reinforcement Learning

Tested with Ubuntu 16.04 and Tensorflow r1.11

#### Requirements
```
sudo apt-get install python3-tk
```

```
pip3 install gym atari_py coloredlogs
    termcolor pyglet tables matplotlib
    numpy opencv-python moviepy scipy
    scikit-image pygame
```

## Human Demonstration

#### Using the Collected Human Demonstration
To use the existing human demonstration data, clone the [`atari_human_demo`](https://github.com/gabrieledcjr/atari_human_demo) repo using the code below. This will create the `collected_demo` folder in this repo.
```
git clone https://github.com/gabrieledcjr/atari_human_demo.git collected_demo
```

#### Collect Human Demonstration on Atari (OpenAI Gym)
The following collect will collect human demonstration for the game of `Pong` with 5 episodes where each episode ends when the game is lost or when the time limit of 20 minutes elapses, whichever comes first. Collected data will be saved in the `collected_demo` folder and will be added to the database for the game played.
```
python3 tools/get_demo.py --gym-env=MsPacmanNoFrameskip-v4 --num-episodes=5 --demo-time-limit=20
```


## Supervised Pre-Training (Demo Classification)
Current setting will pre-train a network `MsPacman` using cuda device 0 that uses only 80% of the GPU resources. This will train a model for 750,000 training iterations with a mini-batch size of 32. This also uses the proportional batch sampling and loads the demos whose IDs are specified in `--demo-ids`.
```
python3 pretrain/run_experiment.py --gym-env=MsPacmanNoFrameskip-v4 \
    --cuda-devices=0 --gpu-fraction=.8 \
    --classify-demo --use-mnih-2015 --train-max-steps=750000 --batch_size=32 \
    --grad-norm-clip=0.5 \
    --use-batch-proportion --demo-ids=1,2,3,4,5,6,7,8
```

## A3C

#### A3C Baseline
Note: Use `--rmsp-epsilon=1e-4` for `PongNoFrameskip-v4` and `--rmsp-epsilon=1e-5` for the rest of the Atari domain.

The following code will run an A3C baseline training for Breakout using 16 parallel worker (actor) threads for 100 million training steps using the Feed-Forward network (no LSTM). To adjust total training steps to half, use `--max-time-step-fraction=0.5`.
```
python3 a3c/run_experiment.py --gym-env=MsPacmanNoFrameskip-v4 \
    --parallel-size=16 \
    --max-time-step-fraction=1.0 \
    --initial-learn-rate=0.0007 --rmsp-epsilon=1e-5 --grad-norm-clip=0.5 \
    --use-mnih-2015
    --append-experiment-num=1
```

#### A3C using Transformed Bellman Operator (A3C-TB)
```
python3 a3c/run_experiment.py --gym-env=MsPacmanNoFrameskip-v4 \
    --parallel-size=16 \
    --max-time-step-fraction=1.0 \
    --initial-learn-rate=0.0007 --rmsp-epsilon=1e-5 --grad-norm-clip=0.5 \
    --use-mnih-2015
    --unclipped-reward --transformed-bellman \
    --append-experiment-num=1
```

#### Using a Pre-Trained Network in A3C-TB
If you followed the settings above to generate a supervised pre-trained model, then using the `--use-transfer` argument should find and load the pre-trained model.
```
python3 a3c/run_experiment.py --gym-env=MsPacmanNoFrameskip-v4 \
    --parallel-size=16 \
    --max-time-step-fraction=1.0 \
    --initial-learn-rate=0.0007 --rmsp-epsilon=1e-5 --grad-norm-clip=0.5 \
    --use-mnih-2015 \
    --unclipped-reward --transformed-bellman \
    --use-transfer \
    --append-experiment-num=1
```

## Grad-CAM

#### Generate Agent Demo and Grad-CAM from Final Model of the Pre-Trained A3C-TB
```
python3 a3c/run_experiment.py --gym-env=PongNoFrameskip-v4
    --test-model --eval-max-steps=5000
    --initial-learn-rate=0.0007 --rmsp-epsilon=1e-5 --grad-norm-clip=0.5 --use-mnih-2015
    --folder=results/a3c/PongNoFrameskip_v4_mnih2015_rawreward_transformedbell_transfer_1
    --use-grad-cam
```

## DQN

#### DQN Baseline
```
python3 dqn/run_experiment.py --gym-env=PongNoFrameskip-v4 \
    --cuda-devices=0 --gpu-fraction=1. \
    --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.00001
```
