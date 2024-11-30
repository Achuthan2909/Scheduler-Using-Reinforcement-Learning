# examples/multi_rating_basic.py

import numpy as np
import pandas as pd
import random
from worker_scheduler.environment.multi_rating_env import MultiRatingEnvironment
from worker_scheduler.agents.multi_rating_agent import MultiRatingAgent
from worker_scheduler.training.multi_rating_train import train_multi_rating_scheduler
# from worker_scheduler.utils.multi_rating_visualization import analyze_assignments
# from worker_scheduler.utils.multi_rating_logger import MultiRatingLogger
from worker_scheduler.utils.multi_rating_export_utils import export_assignments_to_csv
from worker_scheduler.environment.env import WorkerSchedulingEnv
from worker_scheduler.training.train import run_training

def create_controller_profiles(num_controllers):
    """Create sample controller profiles with multiple ratings per controller"""
    all_positions = [f'r{i}' for i in range(1, 9)]  # r1 through r8
    
    profiles_data = []
    for i in range(num_controllers):
        # Randomly select 2-4 positions for each controller
        num_ratings = random.randint(2, 4)
        # Ensure ratings make sense for each controller
        ratings = set(random.sample(all_positions, num_ratings))
        
        profiles_data.append({
            'Controller_ID': f'CTL{i+1:03d}',
            'Ratings': ratings
        })
    
    return pd.DataFrame(profiles_data).set_index('Controller_ID')

def main():
    # Configuration
    num_controllers = 300
    num_shifts = int(input("Enter the number of shifts for the station: "))
    num_teams = num_shifts+2
    random.seed(42)  # For reproducibility
    
    print("Creating controller profiles...")
    controller_profiles = create_controller_profiles(num_controllers)
    
    # Display some profile statistics
    ratings_per_controller = controller_profiles['Ratings'].apply(len)
    print("\nController Profile Statistics:")
    print(f"Total Controllers: {num_controllers}")
    print(f"Average Ratings per Controller: {ratings_per_controller.mean():.2f}")
    print(f"Min Ratings: {ratings_per_controller.min()}")
    print(f"Max Ratings: {ratings_per_controller.max()}")
    
    # Team requirements - each team needs various positions
    val = []
    for i in range(8):
        values = int(input("Enter the minimum number of people required in position r"f'{i}'":" ))
        val.append(values)

    # Get distinct values and sort them
    distinct_requirements = sorted(set(val))
    print("\nDistinct worker requirements:", distinct_requirements)

    # Get shift durations
    shift_durations = []
    for i in range(num_shifts):
        duration = int(input(f"Enter duration for shift {i+1} (in minutes): "))
        shift_durations.append(duration)

    team_requirements = [
        {
            'r1': val[0]*2, 'r2': val[1]*2, 'r3': val[2]*2, 'r4': val[3]*2,
            'r5': val[4]*2, 'r6': val[5]*2, 'r7': val[6]*2, 'r8': val[7]*2
        }
        for _ in range(num_teams)
    ]
    
    # Print team requirements
    print("\nTeam Requirements:")
    for team_idx, reqs in enumerate(team_requirements):
        print(f"\nTeam {team_idx + 1}:")
        for pos, count in reqs.items():
            print(f"  {pos}: {count}")
    
    # Initialize environment and agent
    print("\nInitializing environment and agent...")
    env = MultiRatingEnvironment(controller_profiles, team_requirements)
    agent = MultiRatingAgent(env)
    
    # Train the agent
    print("\nStarting training process...")
    best_assignments, episode_rewards = train_multi_rating_scheduler(
        env, agent, num_episodes=1000
    )
    
    # Validate and analyze results
    if best_assignments:
        # print("\nValidating assignments...")
        # validation_metrics = validate_assignments(best_assignments, env)
        
        print("\nAnalyzing results...")
        # Analyze and visualize results
        # analyze_assignments(best_assignments, controller_profiles, num_teams, episode_rewards)
        
        # Print detailed assignment summary
        # MultiRatingLogger.print_assignment_summary(
        #     best_assignments, 
        #     controller_profiles, 
        #     team_requirements
        # )
        
        # Export results to CSV
        export_assignments_to_csv(best_assignments, controller_profiles)
        
        # Print final validation metrics
        # print("\nFinal Validation Metrics:")
        # print(f"Total Assignments: {validation_metrics['total_assignments']}")
        # print(f"Valid Rating Assignments: {validation_metrics['valid_ratings']}")
        # print(f"Valid Rating Percentage: {validation_metrics['valid_rating_percentage']:.1f}%")
        
        # if validation_metrics['unfilled_positions']:
        #     print("\nUnfilled Positions:")
        #     for team, pos, count in validation_metrics['unfilled_positions']:
        #         print(f"Team {team + 1}, Position {pos}: Missing {count}")
    else:
        print("No valid assignments found!")

    # Run PPO for each distinct requirement and shift duration
    for req in distinct_requirements:
        if req !=0:
            for shift_duration in shift_durations:
                env = WorkerSchedulingEnv(
                    n_workers=req*2,  # max workers is double the requirement
                    min_work_time=60,
                    max_work_time=90,
                    total_shift_duration=shift_duration,
                    min_workers_required=req
                )
                ppo, metrics = run_training(env=env, save_dir=f'./outputs/req_{req}_shift_{shift_duration}')
        else:
            continue

if __name__ == "__main__":
    main()