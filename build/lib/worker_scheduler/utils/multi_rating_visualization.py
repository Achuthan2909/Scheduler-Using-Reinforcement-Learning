# src/work_scheduler/utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def analyze_assignments(assignments, controller_profiles, num_teams, episode_rewards=None):
    """Comprehensive analysis of assignments using matplotlib"""
    if not assignments:
        print("No assignments to analyze!")
        return
    
    # Set up the figure with subplots
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 25))
    
    # 1. Rating Distribution per Team
    ax1 = plt.subplot(4, 2, 1)
    team_ratings = defaultdict(lambda: defaultdict(int))
    for controller_id, team, pos in assignments:
        team_ratings[f'Team {team+1}'][pos] += 1
    
    df_ratings = pd.DataFrame(team_ratings).fillna(0)
    x = np.arange(len(df_ratings.index))
    width = 0.15
    
    for i, team in enumerate(df_ratings.columns):
        ax1.bar(x + i * width, df_ratings[team], width, label=team)
    
    ax1.set_title('Position Distribution by Team', pad=20)
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Number of Controllers')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(df_ratings.index, rotation=45)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Team Size Distribution
    ax2 = plt.subplot(4, 2, 2)
    team_sizes = defaultdict(int)
    for _, team, _ in assignments:
        team_sizes[f'Team {team + 1}'] += 1
    
    teams = list(team_sizes.keys())
    sizes = list(team_sizes.values())
    bars = ax2.bar(teams, sizes, color='skyblue')
    ax2.set_title('Team Size Distribution', pad=20)
    ax2.set_xlabel('Team')
    ax2.set_ylabel('Number of Controllers')
    ax2.axhline(y=np.mean(sizes), color='r', linestyle='--', label='Mean Size')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Invalid Assignment Matrix
    ax3 = plt.subplot(4, 2, 3)
    invalid_matrix = np.zeros((num_teams, 8))
    for controller_id, team, pos in assignments:
        pos_level = int(pos[1])
        if pos not in controller_profiles.loc[controller_id, 'Ratings']:
            invalid_matrix[team][pos_level-1] += 1
    
    im = ax3.imshow(invalid_matrix, cmap='YlOrRd')
    ax3.set_xticks(np.arange(8))
    ax3.set_yticks(np.arange(num_teams))
    ax3.set_xticklabels([f'R{i+1}' for i in range(8)])
    ax3.set_yticklabels([f'Team {i+1}' for i in range(num_teams)])
    ax3.set_title('Invalid Assignments Heatmap', pad=20)
    
    plt.colorbar(im, ax=ax3)
    
    # Add value annotations
    for i in range(num_teams):
        for j in range(8):
            text = ax3.text(j, i, int(invalid_matrix[i, j]),
                        ha="center", va="center", 
                        color="black" if invalid_matrix[i, j] < np.max(invalid_matrix)/2 else "white")
    
    # 4. Learning Progress
    ax4 = plt.subplot(4, 2, 4)
    if episode_rewards is not None:
        window_size = 50
        rolling_mean = pd.Series(episode_rewards).rolling(window=window_size).mean()
        ax4.plot(episode_rewards, alpha=0.3, label='Episode Rewards', color='gray')
        ax4.plot(rolling_mean, label=f'{window_size}-Episode Moving Average', 
                color='blue', linewidth=2)
        ax4.set_title('Learning Progress', pad=20)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Reward')
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.7)
    
    # 5. Position Fill Rates
    ax5 = plt.subplot(4, 2, 5)
    team_positions = defaultdict(lambda: defaultdict(int))
    for _, team, pos in assignments:
        team_positions[f'Team {team+1}'][pos] += 1
    
    df_positions = pd.DataFrame(team_positions).fillna(0)
    x = np.arange(len(df_positions.index))
    width = 0.15
    
    for i, team in enumerate(df_positions.columns):
        ax5.bar(x + i * width, df_positions[team], width, label=team)
    
    ax5.set_title('Position Fill Rates by Team', pad=20)
    ax5.set_xlabel('Position')
    ax5.set_ylabel('Number of Controllers')
    ax5.set_xticks(x + width * 2)
    ax5.set_xticklabels(df_positions.index, rotation=45)
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.7)
    
    # 6. Valid vs Invalid Assignments
    ax6 = plt.subplot(4, 2, 6)
    rating_pos_matrix = np.zeros((2, 8))  # 2 rows: valid and invalid
    for controller_id, _, pos in assignments:
        pos_level = int(pos[1]) - 1
        if pos in controller_profiles.loc[controller_id, 'Ratings']:
            rating_pos_matrix[0][pos_level] += 1  # Valid
        else:
            rating_pos_matrix[1][pos_level] += 1  # Invalid
    
    x = np.arange(8)
    width = 0.35
    
    ax6.bar(x - width/2, rating_pos_matrix[0], width, label='Valid', color='lightgreen')
    ax6.bar(x + width/2, rating_pos_matrix[1], width, label='Invalid', color='lightcoral')
    
    ax6.set_title('Valid vs Invalid Assignments by Position', pad=20)
    ax6.set_xlabel('Position')
    ax6.set_ylabel('Number of Assignments')
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'R{i+1}' for i in range(8)])
    ax6.legend()
    ax6.grid(True, linestyle='--', alpha=0.7)
    
    # 7. Cumulative Reward Progress
    ax7 = plt.subplot(4, 2, 7)
    if episode_rewards is not None:
        cumulative_rewards = np.cumsum(episode_rewards)
        ax7.plot(cumulative_rewards, color='green')
        ax7.set_title('Cumulative Reward Progress', pad=20)
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('Cumulative Reward')
        ax7.grid(True, linestyle='--', alpha=0.7)
    
    # 8. Assignment Quality Distribution
    ax8 = plt.subplot(4, 2, 8)
    
    # Calculate metrics
    total_assignments = len(assignments)
    valid_assignments = sum(1 for c, _, p in assignments 
                          if p in controller_profiles.loc[c, 'Ratings'])
    invalid_assignments = total_assignments - valid_assignments
    
    metrics = ['Valid\nAssignments', 'Invalid\nAssignments']
    values = [
        (valid_assignments/total_assignments) * 100,
        (invalid_assignments/total_assignments) * 100,
    ]
    
    colors = ['lightgreen', 'lightcoral']
    wedges, texts, autotexts = ax8.pie(values, labels=metrics, colors=colors, 
                                    autopct='%1.1f%%', 
                                    shadow=True, startangle=90)
    ax8.set_title('Assignment Quality Distribution', pad=20)
    
    plt.setp(autotexts, size=8, weight="bold")
    plt.setp(texts, size=8)
    
    plt.tight_layout(pad=3.0)
    plt.show()
    
    # Print Statistical Summary
    print("\nStatistical Summary:")
    print("-" * 50)
    print(f"\nTotal Assignments: {total_assignments}")
    print(f"Valid Assignments: {valid_assignments} ({(valid_assignments/total_assignments)*100:.1f}%)")
    print(f"Invalid Assignments: {invalid_assignments} ({(invalid_assignments/total_assignments)*100:.1f}%)")
    print(f"\nTeam Size Standard Deviation: {np.std(list(team_sizes.values())):.2f}")
    
    return {
        'team_sizes': team_sizes,
        'valid_assignments': valid_assignments,
        'invalid_assignments': invalid_assignments,
        'team_positions': team_positions
    }