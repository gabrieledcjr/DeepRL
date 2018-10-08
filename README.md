# DeepRL

Install `common` package
```
pip3 install --user .
```

Collect human demonstration using OpenAI Gym Atari domain
```
python3 collect_demo/get_demo.py --gym-env=PongNoFrameskip-v4 --num-episodes=5 --demo-time-limit=5
```
