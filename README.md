# TennisGym

![Trained Agent](https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent")

Train a Deep Deterministic Policy Gradient (DDPG) agent to play tennis in the Unity ML-Agents Tennis environment.

## The environment

Two agents control rackets to bounce a ball over a net. Each racket observes 8 variables (the position and velocity of the ball and racket) and takes 2 continuous actions (movement toward or away from the net, and jumping).

Reward scheme:

- +0.1 when the agent hits the ball over the net
- -0.01 when the agent lets the ball hit the ground or hits it out of bounds

The goal is to keep the ball in play. The environment is considered solved when the average score reaches +0.5, where a per-episode score is the maximum over the two agents of their undiscounted summed rewards, averaged over 100 consecutive episodes.

## Approach

A single shared actor-critic pair controls both rackets (self-play). Both agents write their transitions into one shared replay buffer, and the shared policy and value networks are trained on batches sampled from it. Exploration uses an Ornstein-Uhlenbeck noise process, with soft target updates, batch normalization, L2 weight decay, and gradient clipping to stabilize training. Details and training curves are in [Report.md](Report.md).

## Repo structure

- `Tennis.ipynb` - main notebook: explore the environment, train the agent, watch the trained agent play
- `notebook/Tennis.ipynb` - earlier draft of the training notebook
- `Soccer.ipynb` - starter template for the related Unity Soccer task (kept for reference)
- `scripts/` - training code: `agent.py` (shared DDPG agent), `model.py` (actor and critic networks), `ReplayBuffer.py` (shared replay buffer)
- `model/` - trained actor and critic weights
- `gym/` - Unity Tennis environment binary (macOS build)
- `snapshot/` - training score plot and a video of the trained agent
- `Report.md` - write-up of the approach and results

## How to train

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Training requires the Unity ML-Agents Tennis environment. A macOS build is included at `gym/Tennis.app`. Builds for other platforms are available at https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip

3. Open `Tennis.ipynb` and run the training section. The agent is considered solved once the 100-episode average score reaches +0.5; checkpoints are saved to `model/` each episode.

To watch a trained agent play, run the "Watch the trained agent" section of the notebook, which loads the weights from `model/`.

## What you learn

This project shows how a continuous-control DDPG agent is built and trained: actor and critic networks with target networks and soft updates, a shared replay buffer for self-play with two agents, Ornstein-Uhlenbeck exploration noise, and the standard stabilization tricks (batch norm, gradient clipping, weight decay) needed to make it converge on the Unity Tennis task.
