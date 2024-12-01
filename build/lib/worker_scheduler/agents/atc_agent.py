import random
import numpy as np

class ATCAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.05):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.q_table = {}
        for atc_id in env.atc_profiles.index:
            for team in range(env.num_teams):
                for pos in range(1, env.num_positions + 1):
                    self.q_table[(atc_id, team, f'r{pos}')] = random.uniform(-0.1, 0.1)

    def choose_action(self, atc_id, valid_actions, current_assignments, training=True):
        """Choose action using epsilon-greedy strategy"""
        if not valid_actions:
            return None
            
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
            
        state_features = self.env.get_state_features(atc_id, current_assignments)
        q_values = []
        
        for team, pos in valid_actions:
            q_value = self.q_table[(atc_id, team, pos)]
            # Team balance bonus
            team_sizes = [sum(1 for _, t, _ in current_assignments if t == i) 
                         for i in range(self.env.num_teams)]
            if team_sizes:
                min_team_size = min(team_sizes)
                if sum(1 for _, t, _ in current_assignments if t == team) == min_team_size:
                    q_value += 0.1
            q_values.append(q_value)
            
        max_q = max(q_values)
        best_actions = [action for action, q in zip(valid_actions, q_values) 
                       if abs(q - max_q) < 1e-6]
        return random.choice(best_actions)

    def update_q_value(self, atc_id, action, reward, next_atc, next_actions):
        """Update Q-value for the given state-action pair"""
        if next_atc and next_actions:
            max_next_q = max(self.q_table[(next_atc, t, p)] for t, p in next_actions)
        else:
            max_next_q = 0
            
        team, position = action
        current_q = self.q_table[(atc_id, team, position)]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[(atc_id, team, position)] = new_q

    def decay_epsilon(self):
        """Decay epsilon value"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)