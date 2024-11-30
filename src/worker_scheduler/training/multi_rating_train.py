# src/work_scheduler/training/multi_rating_train.py

import random
import numpy as np
from collections import defaultdict

def train_multi_rating_scheduler(env, agent, num_episodes=1000):
    """
    Training loop for multi-rating controller scheduler
    
    Parameters:
    -----------
    env : MultiRatingEnvironment
        Environment instance
    agent : MultiRatingAgent
        Agent instance
    num_episodes : int
        Number of training episodes
        
    Returns:
    --------
    tuple : (best_assignments, episode_rewards)
        best_assignments: List of best (controller_id, team, position) assignments
        episode_rewards: List of rewards per episode
    """
    best_assignments = None
    best_reward = float('-inf')
    episode_rewards = []
    window_size = 100
    running_rewards = []
    
    print("\nStarting training...")
    for episode in range(num_episodes):
        current_assignments = []
        episode_reward = 0
        
        # Sort controllers by number of ratings (most flexible first)
        sorted_controllers = sorted(
            env.controller_profiles.index,
            key=lambda x: len(env.controller_profiles.loc[x, 'Ratings']),
            reverse=True
        )
        
        for i, controller_id in enumerate(sorted_controllers):
            valid_actions = env.get_valid_actions(controller_id, current_assignments)
            if not valid_actions:
                continue
                
            # Choose and perform action
            action = agent.choose_action(controller_id, valid_actions, current_assignments)
            if not action:
                continue
                
            team, position = action
            current_assignments.append((controller_id, team, position))
            
            # Calculate reward
            step_reward = env.calculate_reward(current_assignments)
            episode_reward += step_reward
            
            # Q-learning update
            next_controller = sorted_controllers[i + 1] if i + 1 < len(sorted_controllers) else None
            next_actions = []
            if next_controller:
                next_actions = env.get_valid_actions(next_controller, current_assignments)
            
            agent.update_q_value(controller_id, action, step_reward, next_controller, next_actions)
        
        # Track progress
        episode_rewards.append(episode_reward)
        running_rewards.append(episode_reward)
        if len(running_rewards) > window_size:
            running_rewards.pop(0)
        
        # Update best solution
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_assignments = current_assignments.copy()
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(running_rewards)
            max_reward = max(running_rewards)
            # print(f"Episode {episode + 1}, "
            #       f"Avg Reward: {avg_reward:.2f}, "
            #       f"Max Reward: {max_reward:.2f}, "
            #       f"Epsilon: {agent.epsilon:.3f}")
            
            # Print additional metrics
            if best_assignments:
                total = len(best_assignments)
                matched = sum(1 for c, _, p in best_assignments 
                            if p in env.controller_profiles.loc[c, 'Ratings'])
                # print(f"Current Best Solution - "
                #       f"Total Assignments: {total}, "
                #       f"Valid Ratings: {matched} ({(matched/total)*100:.1f}%)")
    
    return best_assignments, episode_rewards

def validate_assignments(assignments, env):
    """
    Validate the final assignments
    
    Parameters:
    -----------
    assignments : list
        List of (controller_id, team, position) assignments
    env : MultiRatingEnvironment
        Environment instance
        
    Returns:
    --------
    dict : Dictionary of validation metrics
    """
    if not assignments:
        return {"valid": False, "message": "No assignments provided"}
    
    # Track assignments per team
    team_counts = defaultdict(lambda: defaultdict(int))
    validation_metrics = {
        "total_assignments": len(assignments),
        "valid_ratings": 0,
        "team_sizes": defaultdict(int),
        "position_fill_rates": defaultdict(lambda: defaultdict(int)),
        "unfilled_positions": []
    }
    
    # Check each assignment
    for controller_id, team, position in assignments:
        # Count team assignments
        validation_metrics["team_sizes"][team] += 1
        validation_metrics["position_fill_rates"][team][position] += 1
        team_counts[team][position] += 1
        
        # Check if controller is rated for position
        if position in env.controller_profiles.loc[controller_id, 'Ratings']:
            validation_metrics["valid_ratings"] += 1
            
    # Check position requirements
    for team in range(env.num_teams):
        for pos, required in env.team_requirements[team].items():
            current = team_counts[team][pos]
            if current < required:
                validation_metrics["unfilled_positions"].append(
                    (team, pos, required - current)
                )
    
    validation_metrics["valid_rating_percentage"] = (
        validation_metrics["valid_ratings"] / validation_metrics["total_assignments"] * 100
    )
    
    return validation_metrics