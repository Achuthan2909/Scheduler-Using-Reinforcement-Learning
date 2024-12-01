
import numpy as np
import pandas as pd
from collections import defaultdict

class MultiRatingEnvironment:
    def __init__(self, controller_profiles, team_requirements):
        """
        Initialize the environment with controller profiles and team requirements
        
        Parameters:
        -----------
        controller_profiles : pandas.DataFrame
            DataFrame with columns:
            - 'Controller_ID': unique identifier
            - 'Ratings': list/set of ratings the controller is qualified for
        team_requirements : list of dict
            List of dictionaries specifying required positions for each team
            Example: [{'r1': 2, 'r2': 3, 'r3': 2, ...}, ...]
        """
        self.controller_profiles = controller_profiles
        self.team_requirements = team_requirements
        self.num_teams = len(team_requirements)
        self.num_positions = 8  # r1 through r8
        
    def get_valid_positions(self, controller_id):
        """Get all positions a controller can work based on their ratings"""
        return self.controller_profiles.loc[controller_id, 'Ratings']

    def get_valid_actions(self, controller_id, current_assignments):
        """Get all valid team/position combinations for a controller"""
        valid_actions = []
        possible_positions = self.get_valid_positions(controller_id)
        
        # Track current assignments per team
        team_counts = defaultdict(lambda: defaultdict(int))
        for _, team, pos in current_assignments:
            team_counts[team][pos] += 1
        
        # Check each possible team/position combination
        for team in range(self.num_teams):
            for pos in possible_positions:
                # Check if position limit reached for team
                if team_counts[team][pos] < self.team_requirements[team][pos]:
                    valid_actions.append((team, pos))
        
        return valid_actions

    def calculate_reward(self, assignments):
        """Calculate reward for current state"""
        reward = 0
        team_counts = defaultdict(lambda: defaultdict(int))
        
        # Count current assignments
        for controller_id, team, pos in assignments:
            team_counts[team][pos] += 1
        
        # Reward components
        position_match_reward = 0  # Reward for using qualified positions
        seat_filling_reward = 0    # Reward for filling required positions
        balance_bonus = 0          # Reward for team balance
        
        # Calculate position match reward
        for controller_id, team, pos in assignments:
            if pos in self.get_valid_positions(controller_id):
                position_match_reward += 100  # Reward for valid position assignment
        
        # Calculate seat filling rewards
        total_seats = 0
        filled_seats = 0
        
        for team in range(self.num_teams):
            for pos, required in self.team_requirements[team].items():
                current = team_counts[team][pos]
                total_seats += required
                filled_seats += min(current, required)
                
                if current >= required:
                    seat_filling_reward += 100 * required  # Full reward for meeting requirements
                else:
                    seat_filling_reward += 50 * current    # Partial reward for partial filling

        # Large bonus for filling all required positions
        if filled_seats == total_seats:
            seat_filling_reward += 1000
        
        # Calculate team balance bonus
        team_sizes = [sum(counts.values()) for counts in team_counts.values()]
        if team_sizes:
            mean_size = np.mean(team_sizes)
            balance_bonus = 100 - sum(abs(size - mean_size) for size in team_sizes) * 10
        
        # Combine all reward components
        reward = (
            position_match_reward * 2.0 +    # Higher weight for valid position assignments
            seat_filling_reward * 1.5 +      # Strong emphasis on filling positions
            balance_bonus * 0.5              # Smaller weight for team balance
        )
        
        return reward

    def get_state_features(self, controller_id, current_assignments):
        """Extract relevant state features for a controller"""
        team_counts = defaultdict(lambda: defaultdict(int))
        for _, team, pos in current_assignments:
            team_counts[team][pos] += 1
            
        features = []
        
        # Add features about team balance
        for team in range(self.num_teams):
            team_size = sum(team_counts[team].values())
            features.append(team_size)
            
        # Add features about position requirements
        for team in range(self.num_teams):
            for pos in ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8']:
                remaining = self.team_requirements[team][pos] - team_counts[team][pos]
                features.append(remaining)
        
        return np.array(features)