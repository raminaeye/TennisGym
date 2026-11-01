# TennisGym

A DDPG agent in PyTorch that learns to play tennis against itself in the Unity ML-Agents Tennis environment, with one shared actor-critic for both rackets. Built for the Udacity Deep RL Nanodegree.

## What's inside

- `Tennis.ipynb` trains and runs the agent
- `scripts/` holds the agent, networks and replay buffer
- `Report.md` explains the approach

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Tennis.ipynb
```

Use Python 3.6, which the `unityagents` package needs.

## Related tutorial

- [Multi-Agent Reinforcement Learning](https://raminaeye.github.io/multi-agent-rl.html)
