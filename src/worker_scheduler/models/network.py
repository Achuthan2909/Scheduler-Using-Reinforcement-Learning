import torch
import torch.nn as nn

class SchedulerNetwork(nn.Module):
    def __init__(self, n_workers, hidden_size=128):
        super().__init__()
        
        # Input size is n_workers * 5 (5 features per worker)
        self.n_workers = n_workers
        input_size = n_workers * 5
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Action head (output size is 2 for each worker: break or work)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * n_workers)  # 2 actions per worker
        )
        
        # Value head
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        # Ensure input is properly shaped
        if len(x.shape) == 2:  # If input is (n_workers, 5)
            x = x.reshape(-1, self.n_workers * 5)
        elif len(x.shape) == 3:  # If input is batched (batch, n_workers, 5)
            x = x.reshape(x.shape[0], -1)
            
        shared = self.shared(x)
        
        # Get action logits and value
        action_logits = self.actor(shared)
        value = self.critic(shared)
        
        return action_logits, value
