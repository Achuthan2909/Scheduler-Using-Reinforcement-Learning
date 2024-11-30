import os
import pandas as pd

def export_assignments_to_csv(assignments, atc_profiles):
    """
    Export ATC assignments to a CSV file in the outputs folder
    
    Parameters:
    -----------
    assignments : list of tuples
        List of (atc_id, team, position) assignments
    atc_profiles : pandas.DataFrame
        DataFrame containing ATC profiles
    """
    # Create outputs directory if it doesn't exist
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a list to store assignment data
    assignment_data = []
    
    # Process each assignment
    for atc_id, team, position in assignments:
        assignment_data.append({
            'Name': atc_id,
            'Team': f'Team {team + 1}',  # Adding 1 to make teams 1-based
            'Position': position,
            'Rating': atc_profiles.loc[atc_id, 'Rating']
        })
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(assignment_data)
    filename = os.path.join(output_dir, 'teams&position.csv')
    df.to_csv(filename, index=False)
    print(f"\nAssignments saved to {filename}")