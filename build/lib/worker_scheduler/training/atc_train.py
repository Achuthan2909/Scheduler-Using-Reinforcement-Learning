import random
import numpy as np

def train_atc_scheduler(env, agent, num_episodes=1000):
    """Training loop for ATC scheduler"""
    best_assignments = None
    best_reward = float('-inf')
    episode_rewards = []
    window_size = 100
    running_rewards = []
    
    print("\nStarting training...")
    for episode in range(num_episodes):
        current_assignments = []
        episode_reward = 0
        
        # Sort ATCs by rating with random tiebreaker
        sorted_atcs = sorted(
            env.atc_profiles.index,
            key=lambda x: (env.atc_profiles.loc[x, 'Rating'], random.random()),
            reverse=True
        )
        
        for i, atc_id in enumerate(sorted_atcs):
            valid_actions = env.get_valid_actions(atc_id, current_assignments)
            if not valid_actions:
                continue
                
            action = agent.choose_action(atc_id, valid_actions, current_assignments)
            if not action:
                continue
                
            team, position = action
            current_assignments.append((atc_id, team, position))
            
            # Calculate reward
            step_reward = env.calculate_reward(current_assignments)
            episode_reward += step_reward
            
            # Q-learning update
            next_atc = sorted_atcs[i + 1] if i + 1 < len(sorted_atcs) else None
            if next_atc:
                next_actions = env.get_valid_actions(next_atc, current_assignments)
            else:
                next_actions = []
            
            agent.update_q_value(atc_id, action, step_reward, next_atc, next_actions)
        
        # Track progress
        episode_rewards.append(episode_reward)
        running_rewards.append(episode_reward)
        if len(running_rewards) > window_size:
            running_rewards.pop(0)
        
        # Update best solution
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_assignments = current_assignments.copy()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(running_rewards)
            max_reward = max(running_rewards)
            print(f"Episode {episode + 1}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Max Reward: {max_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return best_assignments, episode_rewards