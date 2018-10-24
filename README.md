# Leveraging Human Demonstration in Deep Reinforcement Learning (DeepRL)

Install `common` package
```
pip3 install --user .
```

Run DQN Baseline
```
python3 dqn/run_experiment.py --gym-env=PongNoFrameskip-v4 \
    --cuda-devices=0 --gpu-fraction=1. \
    --optimizer=RMS --lr=0.00025 --decay=0.95 --momentum=0.0 --epsilon=0.00001 \
    --train-max-steps=20500000 \
    --folder=networks_rms_dqn
```

Collect human demonstration using OpenAI Gym Atari domain
```
python3 tools/get_demo.py --gym-env=PongNoFrameskip-v4 --num-episodes=5 --demo-time-limit=5
```
