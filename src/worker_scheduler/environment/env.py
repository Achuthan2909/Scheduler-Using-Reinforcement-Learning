from gym import spaces
import gym
import numpy as np

class WorkerSchedulingEnv(gym.Env):
    def __init__(self, 
                 n_workers=2,
                 min_work_time=60,
                 max_work_time=90,
                 total_shift_duration=360,
                 min_workers_required=1,
                 break_time_multiplier=0.2):
        super().__init__()
        
        if min_workers_required > n_workers:
            raise ValueError("Minimum required workers cannot exceed total number of workers")
            
        self.n_workers = n_workers
        self.min_work_time = min_work_time
        self.max_work_time = max_work_time
        self.total_shift_duration = total_shift_duration
        self.min_workers_required = min_workers_required
        self.break_time_multiplier = break_time_multiplier
        
        # Action space: For each worker: 0 (break) or 1 (work)
        self.action_space = spaces.MultiBinary(n_workers)
        
        # State space: For each worker we track:
        # 1. Current status (0: break, 1: working)
        # 2. Time accumulated in current status
        # 3. Total work time in shift
        # 4. Break eligibility (1 if eligible, 0 if not)
        # 5. Time since last break
        self.observation_space = spaces.Box(
            low=0,
            high=float('inf'),
            shape=(n_workers, 5),
            dtype=np.float32
        )
        
        # For rendering: store history of states
        self.state_history = []
        
    def reset(self):
        """Reset environment to initial state"""
        # Initialize all workers to not working
        self.state = np.zeros((self.n_workers, 5), dtype=np.float32)
        
        # Randomly select initial workers
        initial_workers = np.random.choice(
            self.n_workers, 
            size=self.min_workers_required, 
            replace=False
        )
        
        # Set selected workers to working state
        self.state[initial_workers, 0] = 1  # Set status to working (1)
        
        # Initialize all other state variables
        self.state[:, 1] = 0  # Time in current status
        self.state[:, 2] = 0  # Total work time
        self.state[:, 3] = 0  # Break eligibility
        self.state[:, 4] = 0  # Time since last break
        
        self.current_time = 0
        self.state_history = []  # Clear history
        return self.state
    
    def calculate_break_time(self, worked_time):
        """Calculate required break time based on work duration"""
        val = 0.000228124*(worked_time**2.70951)
        break_time = int(val)
        return val
    
    def is_action_valid(self, action, worker_idx):
        """Hard constraint checking for a worker's action"""
        current_status = self.state[worker_idx, 0]  # Current status
        time_in_status = self.state[worker_idx, 1]  # Time in current status
        break_eligible = self.state[worker_idx, 3]   # Break eligibility
        time_since_break = self.state[worker_idx, 4] # Time since last break
        
        proposed_action = action[worker_idx]
        
        # Check minimum workers constraint
        working_count = np.sum(self.state[:, 0] == 1)
        
        # Hard Constraints:
        
        # 1. Cannot interrupt mandatory break
        if current_status == 0:  # On break
            required_break = self.calculate_break_time(time_since_break)
            if time_in_status < required_break and proposed_action == 1:
                return False
                
        # 2. Must work minimum time before break
        if current_status == 1 and proposed_action == 0:  # Working to break
            if not break_eligible:
                return False
                
        # 3. Cannot exceed maximum work time
        if current_status == 1 and proposed_action == 1:
            if time_in_status >= self.max_work_time:
                return False
                
        # 4. Minimum workers requirement
        if current_status == 1 and proposed_action == 0:  # Trying to go on break
            if working_count <= self.min_workers_required:
                return False
        
        return True
    
    def step(self, action):
        reward = 0
        done = False
        info = {}
        
        self.state_history.append(self.state.copy())
        invalid_actions = []
        
        # Calculate initial work distribution for reward calculation
        initial_work = self.state[:, 2].copy()
        initial_std = np.std(initial_work)
        
        for i in range(self.n_workers):
            current_status = self.state[i, 0]
            time_in_status = self.state[i, 1]
            time_since_break = self.state[i, 4]
            
            if not self.is_action_valid(action, i):
                invalid_actions.append(i)
                reward -= 0.5
                
                # Still update time-based counters
                if current_status == 1:
                    self.state[i, 1] += 1
                    self.state[i, 2] += 1
                    self.state[i, 4] += 1
                    self.state[i, 3] = float(time_in_status >= self.min_work_time)
                else:
                    self.state[i, 1] += 1
                continue
            
            # Process valid action
            if action[i] == 1:  # Work
                if current_status == 0:  # Was on break
                    self.state[i, 3] = 0  # Reset break eligibility
                    self.state[i, 4] = 0  # Reset time since break
                
                self.state[i, 0] = 1
                self.state[i, 1] = 0 if current_status != 1 else time_in_status + 1
                self.state[i, 2] += 1
                self.state[i, 4] += 1
                
                reward += 0.4
                
            else:  # Break
                if current_status == 1:  # Was working
                    if self.state[i, 3] == 1:  # Break eligible
                        reward += 1.2
                
                self.state[i, 0] = 0
                self.state[i, 1] = 0 if current_status != 0 else time_in_status + 1
            
            if current_status == 1:
                self.state[i, 3] = float(time_in_status >= self.min_work_time)
        
        # Soft constraints for work distribution
        current_work = self.state[:, 2]
        work_std = np.std(current_work)
        work_mean = np.mean(current_work)
        
        # Penalize uneven distribution
        reward -= work_std * 0.125
        
        # Additional penalty for very uneven distribution
        max_work = np.max(current_work)
        min_work = np.min(current_work)
        if min_work > 0:
            work_ratio = max_work / min_work
            if work_ratio > 1.3:  # More than 30% difference
                reward -= (work_ratio - 1.3)
        
        self.current_time += 1
        
        if work_std < 10:  # Encourage very even distribution
            reward += 1.5
                
        if self.current_time >= self.total_shift_duration:
            done = True
            final_work_std = np.std(self.state[:, 2])
            reward += 100 * (1 - final_work_std / self.total_shift_duration)
        
        info = {
            'working_count': np.sum(self.state[:, 0] == 1),
            'work_std': work_std,
            'total_work': np.sum(current_work),
            'break_count': np.sum(self.state[:, 0] == 0),
            'invalid_actions': invalid_actions,
            'invalid_action_count': len(invalid_actions),
            'work_ratio': max_work/min_work if min_work > 0 else float('inf')
        }
        
        return self.state, reward, done, info
    
    def render(self):
        """Render the environment with history"""
        print(f"\nCurrent Time: {self.current_time}/{self.total_shift_duration}")
        
        # Print current stats
        print("\nCurrent Statistics:")
        working_count = np.sum(self.state[:, 0] == 1)
        on_break = np.sum(self.state[:, 0] == 0)
        print(f"Working: {working_count} workers")
        print(f"On Break: {on_break} workers")
        
        # Calculate the display width based on current time
        timeline_width = min(100, self.total_shift_duration)
        scale_factor = max(1, self.current_time / timeline_width)
        
        # Print timeline markers
        print("\nTimeline:")
        print("Time: ", end="")
        for t in range(0, min(self.current_time + 1, timeline_width + 1), 10):
            actual_time = int(t * scale_factor)
            print(f"{actual_time:3d}".ljust(10), end="")
        print()
        
        # Print timeline
        print("      ┌" + "─" * min(self.current_time, timeline_width) + "┐")
        
        # Character mapping for status
        status_chars = {
            0: "-",   # Break (dark shade)
            1: "█"    # Working (solid block)
        }
        
        # Print each worker's timeline using history
        for i in range(self.n_workers):
            print(f"W{i+1:2d}   │", end="")
            
            # Print historical states
            for t in range(min(self.current_time, timeline_width)):
                history_idx = int(t * scale_factor)
                if history_idx < len(self.state_history):
                    status = int(self.state_history[history_idx][i, 0])
                    print(status_chars[status], end="")
                else:
                    print(" ", end="")
            print("│")
        
        # Print bottom border
        print("      └" + "─" * min(self.current_time, timeline_width) + "┘")
        
        # Print legend
        print("\nLegend:")
        print(f"█ Working   ▓ Break")
        
        # Print detailed worker status
        print("\nDetailed Worker Status:")
        for i in range(self.n_workers):
            print(f"Worker {i+1}:")
            print(f"  Status: {'Working' if self.state[i, 0] == 1 else 'Break'}")
            print(f"  Time in current status: {self.state[i, 1]:.1f}")
            print(f"  Total work time: {self.state[i, 2]:.1f}")
            print(f"  Break eligible: {'Yes' if self.state[i, 3] == 1 else 'No'}")
            print(f"  Time since last break: {self.state[i, 4]:.1f}")
