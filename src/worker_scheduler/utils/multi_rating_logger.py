# src/work_scheduler/utils/multi_rating_logger.py

from collections import defaultdict

class MultiRatingLogger:
    @staticmethod
    def print_assignment_summary(assignments, controller_profiles, team_requirements):
        """Print detailed summary of assignments"""
        if not assignments:
            print("No valid assignments found!")
            return
            
        team_counts = defaultdict(lambda: defaultdict(int))
        invalid_assignments = defaultdict(int)
        
        print("\nAssignment Summary:")
        print("-" * 50)
        
        # Count assignments
        for controller_id, team, pos in assignments:
            team_counts[team][pos] += 1
            
            # Track invalid assignments
            if pos not in controller_profiles.loc[controller_id, 'Ratings']:
                invalid_assignments[team] += 1
        
        # Print team summaries
        for team in range(len(team_requirements)):
            print(f"\nTeam {team + 1}:")
            print("-" * 20)
            
            # Position fill rates
            for pos in sorted(team_requirements[team].keys()):
                current = team_counts[team][pos]
                required = team_requirements[team][pos]
                status = "✓" if current >= required else "✗"
                print(f"  Position {pos}: {current}/{required} {status}")
            
            # Team statistics
            total_positions = sum(team_counts[team].values())
            invalid_count = invalid_assignments[team]
            valid_count = total_positions - invalid_count
            
            print(f"  Total Positions: {total_positions}")
            print(f"  Valid Assignments: {valid_count}")
            print(f"  Invalid Assignments: {invalid_count}")
            if total_positions > 0:
                print(f"  Validity Rate: {(valid_count/total_positions)*100:.1f}%")
        
        # Print detailed assignments
        print("\nDetailed Controller Assignments:")
        print("-" * 50)
        for controller_id, team, pos in sorted(assignments):
            ratings = controller_profiles.loc[controller_id, 'Ratings']
            valid = "✓" if pos in ratings else "✗"
            print(f"{controller_id}: Team {team + 1}, Position {pos} {valid}")
            print(f"  Available Ratings: {sorted(ratings)}")
        
        # Print overall statistics
        total_assignments = len(assignments)
        total_invalid = sum(invalid_assignments.values())
        print("\nOverall Statistics:")
        print("-" * 50)
        print(f"Total Assignments: {total_assignments}")
        print(f"Valid Assignments: {total_assignments - total_invalid}")
        print(f"Invalid Assignments: {total_invalid}")
        print(f"Overall Validity Rate: {((total_assignments - total_invalid)/total_assignments)*100:.1f}%")


