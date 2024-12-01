from collections import defaultdict
import numpy as np
import pandas as pd

class ATCEnvironment:
    def __init__(self, atc_profiles, team_requirements):
        self.atc_profiles = atc_profiles
        self.team_requirements = team_requirements
        self.num_teams = len(team_requirements)
        self.num_positions = 8
        
        # Rating capability hierarchy
        self.rating_hierarchy = {
            'r8': ['r8', 'r7', 'r6', 'r5', 'r4', 'r3'],
            'r7': ['r7', 'r6', 'r5', 'r4', 'r3'],
            'r6': ['r6', 'r5', 'r4', 'r3'],
            'r5': ['r5', 'r4', 'r3'],
            'r4': ['r4', 'r3'],
            'r3': ['r3'],
            'r2': ['r2'],
            'r1': ['r1']
        }

    def get_possible_positions(self, rating):
        """Get all positions an ATC can work based on their rating"""
        return self.rating_hierarchy.get(rating, [rating])

    def get_state_features(self, atc_id, current_assignments):
        """Extract relevant state features for an ATC"""
        team_counts = defaultdict(lambda: defaultdict(int))
        for _, team, pos in current_assignments:
            team_counts[team][pos] += 1
            
        features = []
        rating = self.atc_profiles.loc[atc_id, 'Rating']
        
        # Add features about team balance
        for team in range(self.num_teams):
            team_size = sum(team_counts[team].values())
            features.append(team_size)
            
        # Add features about position requirements
        for team in range(self.num_teams):
            for pos in sorted(self.rating_hierarchy.keys()):
                remaining = self.team_requirements[team][pos] - team_counts[team][pos]
                features.append(remaining)
                
        return np.array(features)

    def get_valid_actions(self, atc_id, current_assignments):
        """Get all valid team/position combinations for an ATC"""
        valid_actions = []
        rating = self.atc_profiles.loc[atc_id, 'Rating']
        possible_positions = self.get_possible_positions(rating)
        
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
        for atc_id, team, pos in assignments:
            team_counts[team][pos] += 1
        
        # Reward components
        rating_reward = 0
        exact_match_bonus = 0
        seat_filling_reward = 0
        balance_bonus = 0
        
        # Calculate rewards
        total_seats = 0
        filled_seats = 0
        
        for atc_id, team, pos in assignments:
            rating = self.atc_profiles.loc[atc_id, 'Rating']
            if rating not in ['r1', 'r2']:
                rating_level = int(rating[1])
                pos_level = int(pos[1])
                
                if rating_level == pos_level:
                    rating_reward += 150
                    exact_match_bonus += 300
                elif pos_level < rating_level:
                    rating_reward += 20 * (rating_level - pos_level)
        
        # Position requirements and seat filling rewards
        for team in range(self.num_teams):
            for pos, required in self.team_requirements[team].items():
                current = team_counts[team][pos]
                total_seats += required
                filled_seats += min(current, required)
                
                if current >= required:
                    seat_filling_reward += 100 * required
                else:
                    seat_filling_reward += 50 * current

        if filled_seats == total_seats:
            seat_filling_reward += 1000
        
        # Team balance bonus
        team_sizes = [sum(counts.values()) for counts in team_counts.values()]
        if team_sizes:
            mean_size = np.mean(team_sizes)
            balance_bonus = 100 - sum(abs(size - mean_size) for size in team_sizes) * 10
        
        # Combine all reward components
        reward = (
            rating_reward * 1.0 +
            exact_match_bonus * 2.0 +
            seat_filling_reward * 1.5 +
            balance_bonus * 0.5
        )
        
        return reward