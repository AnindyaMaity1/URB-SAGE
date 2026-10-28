import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .env import ACTION_NAMES, GreenInfrastructureEnv


class ActorCritic(nn.Module):
    def __init__(self, observation_size, action_count):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(observation_size, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh())
        self.actor = nn.Linear(48, action_count)
        self.critic = nn.Linear(48, 1)

    def forward(self, observation):
        hidden = self.shared(observation)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


class PPOPlanner:
    def __init__(self, seed=42, learning_rate=.002, clip=.2):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = ActorCritic(11, len(ACTION_NAMES))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.clip = clip
        self.reward_history = []

    def _update(self, observations, actions, old_log_probs, returns, advantages):
        obs = torch.tensor(np.asarray(observations), dtype=torch.float32)
        acts = torch.tensor(actions, dtype=torch.long)
        old_log = torch.tensor(old_log_probs, dtype=torch.float32)
        ret = torch.tensor(returns, dtype=torch.float32)
        adv = torch.tensor(advantages, dtype=torch.float32)
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        for _ in range(4):
            logits, values = self.model(obs)
            dist = Categorical(logits=logits)
            ratio = torch.exp(dist.log_prob(acts) - old_log)
            clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
            loss = -torch.minimum(ratio * adv, clipped).mean() + .5 * (values - ret).square().mean() - .01 * dist.entropy().mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), .7)
            self.optimizer.step()

    def train(self, zones, predictions, budget, weights, scenario, episodes=90, green_target=35):
        env = GreenInfrastructureEnv(zones, predictions, budget, weights, scenario, green_target)
        self.reward_history = []
        for _ in range(episodes):
            reset = env.reset()
            observation = reset[0] if isinstance(reset, tuple) else reset
            observations, actions, log_probs, rewards, values = [], [], [], [], []
            done = False
            while not done:
                tensor = torch.tensor(observation, dtype=torch.float32)
                logits, value = self.model(tensor)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                transition = env.step(action.item())
                observation, reward, done = transition[0], transition[1], transition[2]
                observations.append(tensor.numpy())
                actions.append(action.item())
                log_probs.append(distribution.log_prob(action).item())
                rewards.append(reward)
                values.append(value.item())
            returns, advantages = [], []
            gae, next_value = 0.0, 0.0
            for reward, value in zip(reversed(rewards), reversed(values)):
                delta = reward + .96 * next_value - value
                gae = delta + .96 * .92 * gae
                returns.append(gae + value)
                advantages.append(gae)
                next_value = value
            self._update(observations, actions, log_probs, list(reversed(returns)), list(reversed(advantages)))
            self.reward_history.append(float(sum(rewards)))

    def plan(self, zones, predictions, budget, weights, scenario, samples=6, green_target=35):
        best_plan, best_reward = [], -float("inf")
        env = GreenInfrastructureEnv(zones, predictions, budget, weights, scenario, green_target)
        for _ in range(samples):
            reset = env.reset()
            observation = reset[0] if isinstance(reset, tuple) else reset
            done, reward_total = False, 0.0
            plan = []
            while not done:
                with torch.no_grad():
                    logits, _ = self.model(torch.tensor(observation, dtype=torch.float32))
                action = int(torch.argmax(logits).item())
                transition = env.step(action)
                observation, reward, done, info = transition[0], transition[1], transition[2], transition[-1]
                reward_total += reward
                plan = info["plan"]
            if reward_total > best_reward:
                best_reward, best_plan = reward_total, [dict(item) for item in plan]
        return best_plan