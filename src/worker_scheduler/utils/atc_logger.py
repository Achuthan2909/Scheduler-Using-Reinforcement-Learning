# src/work_scheduler/utils/atc_logger.py
from collections import defaultdict

class ATCLogger:
    @staticmethod
    def print_assignment_summary(assignments, atc_profiles, team_requirements):
        """Print summary of assignments"""
        if not assignments:
            print("No valid assignments found!")
            return
            
        team_counts = defaultdict(lambda: defaultdict(int))
        below_rating_assignments = defaultdict(int)
        
        print("\nAssignment Summary:")
        print("-----------------")
        
        # Count assignments
        for atc_id, team, pos in assignments:
            team_counts[team][pos] += 1
            
            # Track below-rating assignments
            rating = atc_profiles.loc[atc_id, 'Rating']
            if rating not in ['r1', 'r2']:
                rating_level = int(rating[1])
                pos_level = int(pos[1])
                if pos_level < rating_level:
                    below_rating_assignments[team] += 1
        
        # Print team summaries
        for team in range(len(team_requirements)):
            print(f"\nTeam {team + 1}:")
            for pos in sorted(team_requirements[team].keys()):
                current = team_counts[team][pos]
                required = team_requirements[team][pos]
                print(f"  {pos}: {current}/{required} (Assigned/Required)")
            print(f"  Total: {sum(team_counts[team].values())}")
            print(f"  Below-rating assignments: {below_rating_assignments[team]}")
            
        # Print detailed assignments
        print("\nDetailed Assignments:")
        print("--------------------")
        for atc_id, team, pos in sorted(assignments):
            rating = atc_profiles.loc[atc_id, 'Rating']
            print(f"{atc_id}: Team {team + 1}, Position {pos} (Rating: {rating})")