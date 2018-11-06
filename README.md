# Leveraging Humans in Deep Reinforcement Learning (DeepRL)

Tested with Ubuntu 16.04 and Tensorflow r1.11

#### Requirements
```
sudo apt-get install python3-tk
```

```
pip3 install gym atari_py coloredlogs
    termcolor pyglet tables matplotlib
    numpy opencv-python moviepy scipy
    scikit-image
```

#### Run DQN Baseline
```
python3 dqn/run_experiment.py --gym-env=PongNoFrameskip-v4 \
    --cuda-devices=0 --gpu-fraction=1. \
    --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.00001
```

#### Run A3C Baseline
Use `--rmsp-epsilon=1e-4` for `PongNoFrameskip-v4` and `--rmsp-epsilon=1e-5` for the rest of the Atari domain.
```
python3 a3c/run_experiment.py --gym-env=BreakoutNoFrameskip-v4 \
    --parallel-size=16 \
    --max-time-step-fraction=1.0 \
    --initial-learn-rate=0.0007 --rmsp-epsilon=1e-4 --grad-norm-clip=0.5 \
    --use-mnih-2015 \
    --use-lstm
```

#### Collect human demonstration using OpenAI Gym Atari domain
```
python3 tools/get_demo.py --gym-env=PongNoFrameskip-v4 --num-episodes=5 --demo-time-limit=5
```
