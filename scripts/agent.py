"""Shared DDPG agent for the Unity Tennis environment (self-play).

A single actor-critic pair is shared by both rackets: each step, both
agents' experiences are added to one shared replay buffer, and the shared
policy and value networks are trained on batches drawn from it.
"""
import numpy as np
import random
import copy
from scripts.model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim
from scripts.ReplayBuffer import ReplayBuffer


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate of the actor
LR_CRITIC = 8e-4        # learning rate of the critic
WEIGHT_DECAY = 0.00001  # L2 weight decay
TIME = 64               # train the networks every TIME time steps
TRAIN_NUM = 128         # number of gradient steps per training round
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment.

    One shared actor-critic pair controls both rackets (self-play). Both
    agents store their transitions in the same replay buffer and the
    networks are updated from samples drawn from it.
    """

    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents sharing this policy (2 for Tennis)
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, 256).to(device)
        self.actor_target = Actor(state_size, action_size, 256).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, 256).to(device)
        self.critic_target = Critic(state_size, action_size, 256).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY
        )

        # Noise process for exploration
        self.noise = OUNoise(self.action_size, mu=0.0, theta=0.1, sigma=0.3)

        # Shared replay memory for all agents
        self.pbuffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE,
                                   random_seed, device)

        # Start target networks as exact copies of the local networks
        self.deep_copy(self.actor_target, self.actor_local)
        self.deep_copy(self.critic_target, self.critic_local)

    def step(self, states, actions, rewards, next_states, dones, time_stamp):
        """Save experiences in replay memory and train on sampled batches.

        Both agents' transitions are stored in the shared buffer. Every
        TIME time steps, TRAIN_NUM gradient updates are run on distinct
        batches (batches already used in this round are skipped) once the
        buffer holds at least BATCH_SIZE experiences.

        Params
        ======
            states (np.ndarray): states for all agents, shape (num_agents, state_size)
            actions (list): actions taken by each agent
            rewards (list): rewards received by each agent
            next_states (np.ndarray): next states for all agents
            dones (list): done flags for each agent
            time_stamp (int): current environment step, used to schedule training
        """
        for i in range(self.num_agents):
            self.pbuffer.add(states[i, :], actions[i], rewards[i],
                             next_states[i, :], dones[i])

        # Learn, if enough samples are available in memory
        seen = []
        if time_stamp % TIME == 0 and len(self.pbuffer) >= BATCH_SIZE:
            for _ in range(TRAIN_NUM):
                p_sample = self.pbuffer.sample()
                batch_mean = p_sample[0].mean().item()
                if batch_mean not in seen:  # encourage distinct batches in one round
                    seen.append(batch_mean)
                    self.learn(p_sample, GAMMA)

    def act(self, states, add_noise=True):
        """Return actions for given states as per the current policy.

        The shared actor is evaluated once per agent (in eval mode so batch
        normalization uses running statistics). Optional Ornstein-Uhlenbeck
        noise encourages exploration during training.

        Params
        ======
            states (np.ndarray): states for all agents, shape (num_agents, state_size)
            add_noise (bool): add exploration noise to the actions
        """
        actions = []
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i, :]).unsqueeze(0).to(device)
            self.actor_local.eval()
            with torch.no_grad():
                action = self.actor_local(state).squeeze(0).cpu().detach().numpy()
                action = np.clip(action, -1, 1)
            self.actor_local.train()
            if add_noise:
                action += self.noise.sample() / 5
            actions.append(action)

        return actions

    def reset(self):
        """Reset the exploration noise process (call at the start of training)."""
        self.noise.reset()

    def learn(self, p_sample, gamma):
        """Update policy and value parameters from a batch of experiences.

        Critic target: Q_targets = r + gamma * critic_target(next_state,
        actor_target(next_state)).

        Params
        ======
            p_sample (Tuple[torch.Tensor]): batch of (states, actions, rewards,
                next_states, dones) sampled from the replay buffer
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = p_sample

        # ---------------------------- update critic ---------------------------- #
        # Best believed action for each next state, from the target actor
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        # Compute critic loss between expected and target Q-values
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.5)
        self.critic_optimizer.step()

        # ----------------------------- update actor ---------------------------- #
        # Maximize expected Q-value: minimize the negative mean Q of the
        # actions the current policy would take.
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1.5)
        self.actor_optimizer.step()

        # ----------------------- soft update target networks ------------------- #
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        theta_target = tau*theta_local + (1 - tau)*theta_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data +
                                    (1.0 - tau) * target_param.data)

    def deep_copy(self, target, source):
        """Copy all parameters from the source model to the target model.

        Params
        ======
            target: PyTorch model (weights will be copied to)
            source: PyTorch model (weights will be copied from)
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise.

    Generates smooth noise that is well suited to continuous control tasks
    with inertia, such as moving a racket.
    """

    def __init__(self, size, mu, theta, sigma):
        """Initialize the noise process.

        Params
        ======
            size (int): dimension of the noise vector (action size)
            mu (float): long-running mean the process reverts to
            theta (float): speed of mean reversion
            sigma (float): volatility of the process
        """
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state to the mean."""
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        """Advance the process one step and return the current noise vector."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state
