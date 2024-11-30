# src/work_scheduler/utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def plot_learning_curve(episode_rewards):
    """Plot the learning curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('Learning Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

def analyze_assignments(assignments, atc_profiles, num_teams, episode_rewards):
    """Comprehensive analysis of assignments using matplotlib"""
    if not assignments:
        print("No assignments to analyze!")
        return
    
    # Set up the figure with subplots
    plt.style.use('default')  # Using default matplotlib style
    fig = plt.figure(figsize=(20, 25))
    
    # 1. Rating Distribution per Team
    ax1 = plt.subplot(4, 2, 1)
    team_ratings = defaultdict(lambda: defaultdict(int))
    for atc_id, team, pos in assignments:
        rating = atc_profiles.loc[atc_id, 'Rating']
        team_ratings[f'Team {team+1}'][rating] += 1
    
    df_ratings = pd.DataFrame(team_ratings).fillna(0)
    x = np.arange(len(df_ratings.index))
    width = 0.15
    
    for i, team in enumerate(df_ratings.columns):
        ax1.bar(x + i * width, df_ratings[team], width, label=team)
    
    ax1.set_title('Rating Distribution by Team', pad=20)
    ax1.set_xlabel('Rating Level')
    ax1.set_ylabel('Number of ATCs')
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
    ax2.set_ylabel('Number of ATCs')
    ax2.axhline(y=np.mean(sizes), color='r', linestyle='--', label='Mean Size')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Below Rating Assignments Matrix
    ax3 = plt.subplot(4, 2, 3)
    below_rating_matrix = np.zeros((num_teams, 8))
    for atc_id, team, pos in assignments:
        rating = atc_profiles.loc[atc_id, 'Rating']
        if rating not in ['r1', 'r2']:
            rating_level = int(rating[1])
            pos_level = int(pos[1])
            if pos_level < rating_level:
                below_rating_matrix[team][pos_level-1] += 1
    
    im = ax3.imshow(below_rating_matrix, cmap='YlOrRd')
    ax3.set_xticks(np.arange(8))
    ax3.set_yticks(np.arange(num_teams))
    ax3.set_xticklabels([f'R{i+1}' for i in range(8)])
    ax3.set_yticklabels([f'Team {i+1}' for i in range(num_teams)])
    ax3.set_title('Below Rating Assignments Heatmap', pad=20)
    
    # Add colorbar
    plt.colorbar(im, ax=ax3)
    
    # Add value annotations
    for i in range(num_teams):
        for j in range(8):
            text = ax3.text(j, i, int(below_rating_matrix[i, j]),
                        ha="center", va="center", 
                        color="black" if below_rating_matrix[i, j] < np.max(below_rating_matrix)/2 else "white")
    
    # 4. Learning Progress
    ax4 = plt.subplot(4, 2, 4)
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
    for atc_id, team, pos in assignments:
        team_positions[f'Team {team+1}'][pos] += 1
    
    df_positions = pd.DataFrame(team_positions).fillna(0)
    x = np.arange(len(df_positions.index))
    width = 0.15
    
    for i, team in enumerate(df_positions.columns):
        ax5.bar(x + i * width, df_positions[team], width, label=team)
    
    ax5.set_title('Position Fill Rates by Team', pad=20)
    ax5.set_xlabel('Position')
    ax5.set_ylabel('Number of ATCs')
    ax5.set_xticks(x + width * 2)
    ax5.set_xticklabels(df_positions.index, rotation=45)
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.7)
    
    # 6. Rating vs Position Distribution
    ax6 = plt.subplot(4, 2, 6)
    rating_pos_matrix = np.zeros((8, 8))
    for atc_id, _, pos in assignments:
        rating = atc_profiles.loc[atc_id, 'Rating']
        if rating not in ['r1', 'r2']:
            rating_level = int(rating[1]) - 1
            pos_level = int(pos[1]) - 1
            rating_pos_matrix[rating_level][pos_level] += 1
    
    im = ax6.imshow(rating_pos_matrix, cmap='viridis')
    ax6.set_xticks(np.arange(8))
    ax6.set_yticks(np.arange(8))
    ax6.set_xticklabels([f'Pos {i+1}' for i in range(8)])
    ax6.set_yticklabels([f'R{i+1}' for i in range(8)])
    ax6.set_title('Rating vs Position Distribution', pad=20)
    plt.colorbar(im, ax=ax6)
    
    # Add value annotations
    for i in range(8):
        for j in range(8):
            text = ax6.text(j, i, int(rating_pos_matrix[i, j]),
                        ha="center", va="center", 
                        color="white" if rating_pos_matrix[i, j] > np.mean(rating_pos_matrix) else "black")
    
    # 7. Cumulative Reward Progress
    ax7 = plt.subplot(4, 2, 7)
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
    below_rating_count = sum(1 for atc_id, _, pos in assignments 
                        if atc_profiles.loc[atc_id, 'Rating'] not in ['r1', 'r2'] 
                        and int(atc_profiles.loc[atc_id, 'Rating'][1]) > int(pos[1]))
    
    optimal_assignments = sum(1 for atc_id, _, pos in assignments 
                            if atc_profiles.loc[atc_id, 'Rating'] not in ['r1', 'r2'] 
                            and int(atc_profiles.loc[atc_id, 'Rating'][1]) == int(pos[1]))
    
    metrics = ['Optimal\nAssignments', 'Below\nRating', 'Other']
    values = [
        (optimal_assignments/total_assignments) * 100,
        (below_rating_count/total_assignments) * 100,
        ((total_assignments - optimal_assignments - below_rating_count)/total_assignments) * 100
    ]
    
    colors = ['lightgreen', 'lightcoral', 'lightgray']
    wedges, texts, autotexts = ax8.pie(values, labels=metrics, colors=colors, 
                                    autopct='%1.1f%%', 
                                    shadow=True, startangle=90)
    ax8.set_title('Assignment Quality Distribution', pad=20)
    
    # Make percentage labels easier to read
    plt.setp(autotexts, size=8, weight="bold")
    plt.setp(texts, size=8)
    
    plt.tight_layout(pad=3.0)
    plt.show()
    
    # Print Statistical Summary
    print("\nStatistical Summary:")
    print("-" * 50)
    print(f"\nTotal Assignments: {total_assignments}")
    print(f"Optimal Assignments: {optimal_assignments} ({(optimal_assignments/total_assignments)*100:.1f}%)")
    print(f"Below Rating Assignments: {below_rating_count} ({(below_rating_count/total_assignments)*100:.1f}%)")
    print(f"\nTeam Size Standard Deviation: {np.std(list(team_sizes.values())):.2f}")
    
    return {
        'team_sizes': team_sizes,
        'below_rating_count': below_rating_count,
        'optimal_assignments': optimal_assignments,
        'team_ratings': team_ratings,
        'team_positions': team_positions,
        'rating_pos_matrix': rating_pos_matrix
    }