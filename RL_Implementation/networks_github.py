import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorNetworkSAC(nn.Module):
    """ Stochastic Gaussian Actor Network for SAC """
    def __init__(self, n_observations: int, n_actions: int, action_high_bound: torch.Tensor | float = 1.0, action_offset: torch.Tensor | float = 0.0, h1 = 32, h2 = 32):
        super(ActorNetworkSAC, self).__init__()
        self.register_buffer("action_high_bound", torch.as_tensor(action_high_bound, dtype=torch.float32))
        self.register_buffer("action_offset", torch.as_tensor(action_offset, dtype=torch.float32))
        # Architecture (adjust complexity as needed)
        self.layer1 = nn.Linear(n_observations, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.mean_layer = nn.Linear(h2, n_actions) # Outputs mean
        self.log_std_layer = nn.Linear(h2, n_actions) # Outputs log standard deviation

    def forward(self, state: torch.Tensor, deterministic = False):
        """
        Outputs action and its log probability, using reparameterization and tanh squashing.
        Parameters:
        - state (torch.Tensor): Input state.
        Returns:
        - Tuple[torch.Tensor, torch.Tensor]:
            - action: Squashed action sampled from the policy.
            - log_prob: Log probability of the squashed action.
        """
        EPSILON = 1e-6
        LOG_STD_MAX = 2
        LOG_STD_MIN = -5

        # Check if state is a single sample and add batch dimension if needed
        add_batch_dim = False
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            add_batch_dim = True

        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))

        mean = self.mean_layer(x)

        if deterministic:
            action = torch.tanh(mean) * self.action_high_bound + self.action_offset
            return action, 0

        log_std = self.log_std_layer(x)
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Create Gaussian distribution
        normal_dist = Normal(mean, std)

        # Reparameterization trick: sample pre-squashed action
        # Use rsample() for differentiable sampling
        z = normal_dist.rsample()

        # Apply tanh squashing to get bounded action
        action = torch.tanh(z)

        # Calculate log-probability with correction for tanh squashing
        # log_prob = log_normal(z) - log(1 - tanh(z)^2)
        log_prob = normal_dist.log_prob(z) - torch.log(1 - action.pow(2) + EPSILON)

        # Sum across action dimensions (proper handling of dimensions)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:
            log_prob = log_prob.sum(keepdim=True)

        # Scale action to environment bounds
        action = action * self.action_high_bound + self.action_offset

        # Remove batch dimension if it was added
        if add_batch_dim:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)

        return action, log_prob
    
class CriticNetworkSAC(nn.Module):
    """ Twin Q-Value Critic Network for SAC """
    def __init__(self, n_observations: int, n_actions: int, h1 = 32, h2 = 32):
        super(CriticNetworkSAC, self).__init__()

        # Q1 Architecture
        self.q1_layer1 = nn.Linear(n_observations + n_actions, h1)
        self.q1_layer2 = nn.Linear(h1, h2)
        self.q1_output = nn.Linear(h2, 1)

        # Q2 Architecture
        self.q2_layer1 = nn.Linear(n_observations + n_actions, h1)
        self.q2_layer2 = nn.Linear(h1, h2)
        self.q2_output = nn.Linear(h2, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Outputs the Q-values from both internal critics.
        Parameters:
        - state (torch.Tensor): Input state tensor.
        - action (torch.Tensor): Input action tensor.
        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Q1(s, a) and Q2(s, a).
        """
        sa = torch.cat([state, action], dim=1) # Concatenate state and action

        # Q1 forward pass
        q1 = F.relu(self.q1_layer1(sa))
        q1 = F.relu(self.q1_layer2(q1))
        q1 = self.q1_output(q1)

        # Q2 forward pass
        q2 = F.relu(self.q2_layer1(sa))
        q2 = F.relu(self.q2_layer2(q2))
        q2 = self.q2_output(q2)

        return q1, q2
