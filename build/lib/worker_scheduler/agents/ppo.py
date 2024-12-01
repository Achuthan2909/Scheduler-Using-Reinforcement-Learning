import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
import os
from datetime import datetime

from ..models.network import SchedulerNetwork
from ..models.memory import Memory
from ..utils.logger import WorkerTimingLogger

class PPO:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, epsilon=0.2, epochs=10):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.gae_lambda = 0.95
        
        self.network = SchedulerNetwork(env.n_workers)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = Memory()
        self.timing_logger = None
        
        # Training metrics
        self.best_reward = float('-inf')
        self.episode_rewards = []
        
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_value_step = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_step = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_step * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        return returns, advantages

    def get_action_and_value(self, state, training=True):
        """Select action using the policy network"""
        state_tensor = torch.FloatTensor(state)
        
        with torch.no_grad():
            logits, value = self.network(state_tensor)
            logits = logits.reshape(-1, self.env.n_workers, 2)
            
            actions = []
            log_probs = []
            
            for worker_logits in logits[0]:
                dist = Categorical(logits=worker_logits)
                if training:
                    action = dist.sample()
                else:
                    action = torch.argmax(worker_logits)
                actions.append(action.item())
                if training:
                    log_probs.append(dist.log_prob(action).item())
        
        return np.array(actions), np.array(log_probs) if training else None, value.item()

    def update_policy(self, states, actions, old_log_probs, returns, advantages):
        """Update policy using PPO loss"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        for _ in range(self.epochs):
            logits, values = self.network(states)
            logits = logits.reshape(-1, self.env.n_workers, 2)
            values = values.squeeze()
            
            new_log_probs = []
            for batch_idx in range(len(states)):
                worker_log_probs = []
                for worker_idx in range(self.env.n_workers):
                    dist = Categorical(logits=logits[batch_idx, worker_idx])
                    worker_log_probs.append(
                        dist.log_prob(actions[batch_idx, worker_idx])
                    )
                new_log_probs.append(torch.stack(worker_log_probs))
            new_log_probs = torch.stack(new_log_probs)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages.unsqueeze(1)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = 0.5 * (returns - values).pow(2).mean()
            
            total_loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

    def save_model(self, path, episode=None, is_best=False):
        """Save model checkpoint"""
        save_dict = {
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'episode': episode,
            'best_reward': self.best_reward,
            'episode_rewards': self.episode_rewards
        }
        
        if is_best:
            save_path = f"{path}/best_model.pt"
        else:
            save_path = f"{path}/checkpoint_episode_{episode}.pt"
        
        torch.save(save_dict, save_path)
        
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.best_reward = checkpoint['best_reward']
        self.episode_rewards = checkpoint['episode_rewards']
        return checkpoint['episode']

    def train(self, max_episodes=10, max_steps=750, batch_size=256, render_freq=50, save_dir='./ouputs'):
        """Main training loop"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"run_{timestamp}")
        os.makedirs(save_path, exist_ok=True)
        
        # Initialize timing logger
        self.timing_logger = WorkerTimingLogger(save_path)
        
        metrics = {
            'episode_rewards': [],
            'avg_rewards': [],
            'episodes': []
        }
        
        for episode in range(max_episodes):
            state = self.env.reset()
            episode_reward = 0
            state_history = []
            
            for step in range(max_steps):
                state_history.append(state.copy())
                
                if episode % render_freq == 0:
                    self.env.render()
                
                action, log_prob, value = self.get_action_and_value(state)
                next_state, reward, done, _ = self.env.step(action)
                
                self.memory.states.append(state)
                self.memory.actions.append(action)
                self.memory.rewards.append(reward)
                self.memory.log_probs.append(log_prob)
                self.memory.values.append(value)
                self.memory.dones.append(done)
                
                state = next_state
                episode_reward += reward
                
                if len(self.memory) >= batch_size:
                    _, _, next_value = self.get_action_and_value(next_state)
                    
                    states_arr = np.array(self.memory.states)
                    actions_arr = np.array(self.memory.actions)
                    log_probs_arr = np.array(self.memory.log_probs)
                    values_arr = np.array(self.memory.values)
                    
                    returns, advantages = self.compute_gae(
                        np.array(self.memory.rewards),
                        values_arr,
                        np.array(self.memory.dones),
                        next_value
                    )
                    
                    self.update_policy(
                        states_arr,
                        actions_arr,
                        log_probs_arr,
                        returns,
                        advantages
                    )
                    
                    self.memory.clear()
                
                if done:
                    break
            
            # Record timing data for the episode
            self.timing_logger.record_episode(episode, state_history, step)
            
            self.episode_rewards.append(episode_reward)
            avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            
            metrics['episode_rewards'].append(episode_reward)
            metrics['avg_rewards'].append(avg_reward)
            metrics['episodes'].append(episode)
            
            np.save(f"{save_path}/metrics.npy", metrics)
            
            if episode % 100 == 0:
                self.save_model(save_path, episode=episode)
            
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.save_model(save_path, episode=episode, is_best=True)
                self.timing_logger.save_excel(episode)
                self.timing_logger.save_gantt(episode)
                print(f"New best model saved! Average Reward: {avg_reward:.2f}")
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Average Reward (last 100): {avg_reward:.2f}")
            
            if avg_reward > 500:
                print("Environment solved!")
                break
        
        self.save_model(save_path, episode=episode)
        return metrics