#!/usr/bin/env python3
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Normal

import numpy as np

from stage_controller_sac.policy_network import PolicyNetwork
from stage_controller_sac.replay_buffer import ReplayBuffer
from stage_controller_sac.q_network import QNetwork

class SAC():
    """A object to train the SAC algorithm."""
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.target_policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_policy.eval()

        self.Q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.Q2 = QNetwork(state_dim, action_dim).to(self.device)
        self.target_Q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.target_Q2 = QNetwork(state_dim, action_dim).to(self.device)
        self.target_Q1.load_state_dict(self.Q1.state_dict())
        self.target_Q2.load_state_dict(self.Q2.state_dict())
        self.target_Q1.eval()
        self.target_Q2.eval()

        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(alpha)).to(self.device)
        self.gamma = gamma
        self.alpha = alpha #self.log_alpha.exp()
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.optimizer_Q1 = optim.Adam(self.Q1.parameters(), lr=learning_rate)
        self.optimizer_Q2 = optim.Adam(self.Q2.parameters(), lr=learning_rate)

    def select_action(self, state):
        """A method to select a action."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy(state)
        action = action.squeeze().cpu().numpy()
        return action

    def update(self, replay_buffer, batch_size):
        """A method to update the current state."""
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.sample_action(next_states)
            next_Q1 = self.target_Q1(next_states, next_actions)
            next_Q2 = self.target_Q2(next_states, next_actions)
            next_Q = torch.min(next_Q1, next_Q2) - self.alpha * next_log_probs
            Q_target = rewards + self.gamma * (1 - dones) * next_Q

        Q1 = self.Q1(states, actions)
        Q2 = self.Q2(states, actions)

        Q1_loss = F.mse_loss(Q1, Q_target)
        Q2_loss = F.mse_loss(Q2, Q_target)
        Q_loss = Q1_loss + Q2_loss

        self.optimizer_Q1.zero_grad()
        self.optimizer_Q2.zero_grad()
        Q_loss.backward()
        self.optimizer_Q1.step()
        self.optimizer_Q2.step()

        actions, log_probs = self.sample_action(states)
        Q1 = self.Q1(states, actions)
        Q2 = self.Q2(states, actions)
        Q = torch.min(Q1, Q2)

        policy_loss = (self.alpha * log_probs - Q).mean()

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

        self.update_target_networks()

    def sample_action(self, state):
        mean = self.policy(state)
        log_std = torch.zeros_like(mean).to(self.device)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = torch.tanh(action)
        return action, log_prob

    def update_target_networks(self):
        tau = 0.005  # Taxa de atualização do target network
        for target_param, param in zip(self.target_Q1.parameters(), self.Q1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_Q2.parameters(), self.Q2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train(self, env, num_episodes, max_steps, batch_size):
        """A method to train the SAC."""
        replay_buffer = ReplayBuffer(capacity=100000)
        total_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0

            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.push(state, action, reward, next_state, done)

                if len(replay_buffer) >= batch_size:
                    self.update(replay_buffer, batch_size)

                state = next_state
                episode_reward += reward

                if done:
                    break

            total_rewards.append(episode_reward)
            avg_reward = np.mean(total_rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes} | \
                  Episode Reward: {episode_reward:.2f} | \
                    Avg Reward: {avg_reward:.2f}")
