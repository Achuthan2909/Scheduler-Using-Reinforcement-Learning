# src/work_scheduler/agents/multi_rating_agent.py

import random
import numpy as np

class MultiRatingAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.05):
        """
        Initialize Q-learning agent for multi-rating controller assignment
        
        Parameters:
        -----------
        env : MultiRatingEnvironment
            The environment the agent will interact with
        learning_rate : float
            Learning rate for Q-learning updates
        discount_factor : float
            Discount factor for future rewards
        epsilon : float
            Initial exploration rate
        epsilon_decay : float
            Rate at which epsilon decays
        epsilon_min : float
            Minimum exploration rate
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.q_table = {}
        for controller_id in env.controller_profiles.index:
            valid_positions = env.get_valid_positions(controller_id)
            for team in range(env.num_teams):
                for pos in valid_positions:
                    self.q_table[(controller_id, team, pos)] = random.uniform(-0.1, 0.1)
    
    def choose_action(self, controller_id, valid_actions, current_assignments, training=True):
        """
        Choose action using epsilon-greedy strategy
        
        Parameters:
        -----------
        controller_id : str
            ID of the controller being assigned
        valid_actions : list
            List of valid (team, position) tuples
        current_assignments : list
            List of current (controller_id, team, position) assignments
        training : bool
            Whether the agent is in training mode
        """
        if not valid_actions:
            return None
            
        # Exploration: random action
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Exploitation: best known action
        state_features = self.env.get_state_features(controller_id, current_assignments)
        q_values = []
        
        for team, pos in valid_actions:
            q_value = self.q_table[(controller_id, team, pos)]
            # Add bonus for balancing teams
            team_sizes = [sum(1 for _, t, _ in current_assignments if t == i) 
                         for i in range(self.env.num_teams)]
            if team_sizes:
                min_team_size = min(team_sizes)
                if sum(1 for _, t, _ in current_assignments if t == team) == min_team_size:
                    q_value += 0.1
            q_values.append(q_value)
        
        # Select best action
        max_q = max(q_values)
        best_actions = [action for action, q in zip(valid_actions, q_values) 
                       if abs(q - max_q) < 1e-6]
        return random.choice(best_actions)
    
    def update_q_value(self, controller_id, action, reward, next_controller, next_actions):
        """
        Update Q-value for the given state-action pair
        
        Parameters:
        -----------
        controller_id : str
            Current controller ID
        action : tuple
            (team, position) tuple
        reward : float
            Reward received
        next_controller : str
            Next controller ID (or None if last controller)
        next_actions : list
            Valid actions for next controller
        """
        if next_controller and next_actions:
            max_next_q = max(self.q_table[(next_controller, t, p)] 
                           for t, p in next_actions)
        else:
            max_next_q = 0
            
        team, position = action
        current_q = self.q_table[(controller_id, team, position)]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[(controller_id, team, position)] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, 
                         self.epsilon * self.epsilon_decay)