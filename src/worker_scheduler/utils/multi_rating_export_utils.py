# src/work_scheduler/utils/export_utils.py

import os
import pandas as pd

def export_assignments_to_csv(assignments, controller_profiles):
    """
    Export controller assignments to a CSV file
    
    Parameters:
    -----------
    assignments : list
        List of (controller_id, team, position) assignments
    controller_profiles : pandas.DataFrame
        DataFrame containing controller profiles with their ratings
    """
    # Create outputs directory if it doesn't exist
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a list to store assignment data
    assignment_data = []
    
    # Process each assignment
    for controller_id, team, pos in assignments:
        ratings = controller_profiles.loc[controller_id, 'Ratings']
        assignment_data.append({
            'Name': controller_id,
            'Team': f'Team {team + 1}',
            'Position': pos,  # The position they're assigned to work
            'Available_Ratings': ', '.join(sorted(ratings)),
        })
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(assignment_data)
    filename = os.path.join(output_dir, 'teams&position.csv')
    df.to_csv(filename, index=False)
    print(f"\nAssignments saved to {filename}")
    