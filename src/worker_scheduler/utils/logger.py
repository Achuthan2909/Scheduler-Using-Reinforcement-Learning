import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

class WorkerTimingLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.timing_data = []
        
    def cleanup_old_files(self, file_pattern):
        """Delete all previous files matching the pattern except the latest one"""
        # Get list of all files matching the pattern
        files = glob.glob(os.path.join(self.save_dir, file_pattern))
        
        # Sort files by creation time
        files.sort(key=os.path.getctime)
        
        # Delete all but the latest file
        for file in files[:-1]:  # All files except the last one
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting file {file}: {e}")

    def record_episode(self, episode_num, state_history, total_duration):
        """Record worker timings for an episode"""
        episode_data = []
        
        for worker_idx in range(len(state_history[0])):
            work_periods = []
            current_period = None
            
            for time_step, state in enumerate(state_history):
                is_working = state[worker_idx][0] == 1
                
                if is_working and current_period is None:
                    current_period = {'start': time_step, 'worker': worker_idx}
                elif not is_working and current_period is not None:
                    current_period['end'] = time_step
                    work_periods.append(current_period)
                    current_period = None
                    
            # Handle case where worker is still working at end
            if current_period is not None:
                current_period['end'] = total_duration
                work_periods.append(current_period)
                
            episode_data.extend(work_periods)
            
        self.timing_data.append({
            'episode': episode_num,
            'periods': episode_data
        })
    
    def save_excel(self, episode_num):
        """Save timing data to Excel file"""
        if not self.timing_data:
            return
            
        latest_data = self.timing_data[-1]
        df_data = []
        
        for period in latest_data['periods']:
            df_data.append({
                'Worker': f"Worker {period['worker'] + 1}",
                'Start Time': period['start'],
                'End Time': period['end'],
                'Duration': period['end'] - period['start']
            })
            
        df = pd.DataFrame(df_data)
        excel_path = os.path.join(self.save_dir, f'worker_timings_episode_{episode_num}.xlsx')
        df.to_excel(excel_path, index=False)
        
        # Cleanup old Excel files
        self.cleanup_old_files('worker_timings_episode_*.xlsx')
        
    def save_gantt(self, episode_num):
        """Save Gantt chart visualization"""
        if not self.timing_data:
            return
            
        latest_data = self.timing_data[-1]
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot each work period
        colors = plt.cm.Set3(np.linspace(0, 1, len(set(p['worker'] for p in latest_data['periods']))))
        
        for period in latest_data['periods']:
            worker_idx = period['worker']
            start = period['start']
            duration = period['end'] - start
            
            ax.barh(f'Worker {worker_idx + 1}', 
                   duration,
                   left=start, 
                   color=colors[worker_idx],
                   alpha=0.8)
                   
        ax.set_xlabel('Time Steps')
        ax.set_title(f'Worker Schedule - Episode {episode_num}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        gantt_path = os.path.join(self.save_dir, f'schedule_gantt_episode_{episode_num}.png')
        plt.savefig(gantt_path)
        plt.close()
        
        # Cleanup old Gantt chart files
        self.cleanup_old_files('schedule_gantt_episode_*.png')