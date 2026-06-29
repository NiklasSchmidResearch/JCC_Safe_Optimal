import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict

from networks_github import ActorNetworkSAC, CriticNetworkSAC

from time import time

class SACAgent:

    """
    Soft actor-critic agent that allows for primal-dual updates for jcc reach-avoid tasks. Can be initialized with any networks. By defualt uses fully connected networks with two hidden layers
    is specified size.
    """

    def __init__(self, nx, nu, action_scale = None, action_offset = None, batch_size = 128, buffer_size = 1e6, alpha_0 = 0.4, target_temp = None,
                 actor = None, critic = None, target_critic = None, a_h1 = 16, a_h2 = 16, c_h1 = 32, c_h2 = 32,
                 a_lr = 1e-4, c_lr = 3e-4, alpha_lr = 1e-4, tau = 1e-3, discount = 0.99, 
                 compile = None, seed = 2701, action_high = None, action_low = None, action_bounds = None):
        
        """
        inputs:
            - nx:               state space dimensions
            - nu:               ction space dimensions
            - action_scale:     Scaling to map interval [-1,1]^n to action box constraints
            - action_offset:    Offset to map interval [-1,1]^n to action box constraints
            - batch_size:       batch size for neural network updats
            - buffer_size:      Max number of transisions to be saved in replay buffer
            - alpha_0:          Initial entropy temp
            - target_temp:      Entropy temperature target value

            - actor:            Optional pre-initialized actor network of any architecture
            - critic:           Optional pre-initialized critic networks of any architecture
            - target_critic:    Optional pre-initialized critic target networks. Should match critic networks

            - a_h1:             Number of neurons in first hidden layer of actor network when using default architecture
            - a_h2:             Number of neurons in second hidden layer of actor network when using default architecture
            - c_h1:             Number of neurons in first hidden layer of critic networks when using default architecture
            - c_h2:             Number of neurons in second hidden layer of critic networks when using default architecture

            - a_lr:             Actor learning rate
            - c_lr:             Critic learning rate
            - alpha_lr:         Entropy temp learning rate
            - tau:              Parameter for soft target updates
            - discount:         Discount factor for mdp

            - compile:          What setting to use to compile pytorch networks
            - seed:             Fixed random seed
            
            - action_high:      Alternative means of specifying action space box constraints
            - action_low:       Alternative means of specifying action space box constraints
            - action_bounds:    Alternative means of specifying action space box constraints

    
        """
        
        # Torch stuff
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.seed = seed
        torch.manual_seed(seed)
        
        # Environment Parameters
        self.nx = nx    # State dim
        self.nu = nu    # Action dim

        action_scale, action_offset = self._action_scale_and_offset(
            nu,
            action_scale=action_scale,
            action_offset=action_offset,
            action_high=action_high,
            action_low=action_low,
            action_bounds=action_bounds,
        )

        # SAC Networks
        self.actor = actor if actor != None else ActorNetworkSAC(nx, nu, 
                    action_high_bound = action_scale.to(self.device), action_offset=action_offset.to(self.device), h1 = a_h1, h2 = a_h2).to(self.device)
        
        self.critic = critic if critic != None else CriticNetworkSAC(nx, nu, c_h1, c_h2).to(self.device)

        self.target_critic = target_critic if target_critic != None else CriticNetworkSAC(self.nx, self.nu, h1 = c_h1, h2 = c_h2).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        for p in self.target_critic.parameters():
            p.requires_grad = False

        # Compile networks
        if compile is not None and self.device == "cuda":
            self.compile = compile
            self.actor.compile(mode = self.compile)
            self.critic.compile(mode = self.compile)
            self.target_critic.compile(mode = self.compile)

        # Entropy
        self.alpha_0 = alpha_0
        self.alpha = alpha_0
        
        if target_temp != None:
            target_temp_tensor = torch.tensor(target_temp, dtype=torch.float32, device=self.device)
            self.target_alpha = target_temp_tensor
        elif target_temp == None:
            self.target_alpha = torch.tensor(-float(nu), dtype=torch.float32, device=self.device)

        self.alpha_lr = alpha_lr

        self.log_alpha = torch.tensor(np.log(self.alpha_0), dtype=torch.float32, requires_grad=True, device=self.device)

        # Optimizers
        self.actor_lr = a_lr
        self.critic_lr = c_lr

        self.actor_optim = optim.Adam(self.actor.parameters(), lr = a_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = c_lr)
        self.alpha_optim = optim.Adam([self.log_alpha], lr = self.alpha_lr)

        # SAC Params
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.discount = discount

        # Other Stuff
        self.replay_buffer = ReplayBuffer(storage = LazyTensorStorage(int(self.buffer_size)), batch_size=batch_size)
        self.actor_loss = 0
        self.critic_loss = 0
        self.alpha_loss = 0
        self.dual_loss = 0

    def _action_scale_and_offset(self, nu, action_scale=None, action_offset=None, action_high=None, action_low=None, action_bounds=None):

        """
        Computes action scaling and offset tensors for mapping the actor's
        normalized output to environment action units.

        inputs:
            - nu:            Action dimension
            - action_scale:  Scalar or length-nu scale applied to normalized actions
            - action_offset: Scalar or length-nu offset added after scaling
            - action_high:   Scalar or length-nu upper action bound
            - action_low:    Scalar or length-nu lower action bound
            - action_bounds: Backward-compatible alias for action_scale

        outputs:
            - action_scale:  Torch tensor scalar or length-nu tensor
            - action_offset: Torch tensor scalar or length-nu tensor
        """

        if action_bounds is not None:
            action_scale = action_bounds
            action_offset = 0.0 if action_offset is None else action_offset

        if action_high is not None or action_low is not None:
            if action_high is None or action_low is None:
                raise ValueError("Provide both action_high and action_low, or neither.")
            high = self._normalize_action_bound(action_high, nu, "action_high")
            low = self._normalize_action_bound(action_low, nu, "action_low")
            action_scale = (high - low) / 2.0
            action_offset = (high + low) / 2.0
        else:
            action_scale = 1.0 if action_scale is None else action_scale
            action_offset = 0.0 if action_offset is None else action_offset
            action_scale = self._normalize_action_bound(action_scale, nu, "action_scale")
            action_offset = self._normalize_action_bound(action_offset, nu, "action_offset")

        return action_scale, action_offset

    def _normalize_action_bound(self, value, nu, name):

        """
        Converts an action bound, scale, or offset value to a float tensor with
        a shape compatible with the action dimension.

        inputs:
            - value: Scalar, length-nu sequence, tensor, or repeated batch of
                     length-nu values
            - nu:    Action dimension
            - name:  Name used in the error message if the shape is invalid

        outputs:
            - tensor: Scalar tensor or length-nu tensor
        """

        tensor = torch.as_tensor(value, dtype=torch.float32)

        if tensor.numel() == 1:
            return tensor

        if tensor.shape == (nu,):
            return tensor

        if tensor.shape[-1] == nu:
            flat = tensor.reshape(-1, nu)
            if torch.allclose(flat, flat[0].expand_as(flat)):
                return flat[0]

        raise ValueError(f"{name} must be scalar or have shape ({nu},); got {tuple(tensor.shape)}.")

    def soft_target_update(self):
        
        """
        Performs soft update of critic target network
        """

        for target_param, main_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)
    
    def sac_update(self, c_update = True, a_update = True, alpha_update = True, dual = 1):

        """
        Performs one SAC update for specified parameters, accounting for value of dual variable

        inputs:
            - c_update:     Flag whether to update critic networks
            - a_update:     Flag whether to update actor network
            - alpha_update: Flag whether to update entropy temperature
            - dual:         Value of dual variable to scale rewards

        outputs:
            - actor_loss
            - critic_loss
            - alpha_loss:   Loss for entropy temp update
            - alpha:        Entropy temp
        """

        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample().to(self.device)

        state_batch = batch["s"]
        action_batch = batch["a"]
        reward_batch = batch["r"] + batch["success"]*dual
        next_state_batch = batch["s'"]
        done_batch = batch["done"]

        self.alpha = torch.exp(self.log_alpha.detach()).float()

        if c_update:
            with torch.no_grad():
                # Get next action and log probability from the current policy
                next_action, next_log_prob = self.actor(next_state_batch)

                # Get target Q values from the target critics
                q1_target_next, q2_target_next = self.target_critic(next_state_batch, next_action)
                q_target_next = torch.min(q1_target_next, q2_target_next)  # Min of two Q-values

                # Calculate the soft target:
                soft_target = q_target_next - self.alpha * next_log_prob

                # Compute the target value for the Bellman equation:
                y = reward_batch.unsqueeze(-1) + self.discount * (1.0 - done_batch.unsqueeze(-1)) * soft_target

            # Get current Q estimates from the critic
            q1_current, q2_current = self.critic(state_batch, action_batch)

            # Calculate critic losses (Mean Squared Error):
            critic1_loss = F.mse_loss(q1_current, y)
            critic2_loss = F.mse_loss(q2_current, y)
            critic_loss = 0.5*(critic1_loss + critic2_loss)

            # Optimize the critic networks
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            self.critic_loss = critic_loss.item()

        if a_update:

            for p in self.critic.parameters():
                p.requires_grad = False

            # Get actions and log probabilities for the current states from the actor
            pi_action, pi_log_prob = self.actor(state_batch)

            # Get Q values for these actions from the critic
            q1_pi, q2_pi = self.critic(state_batch, pi_action)
            min_q_pi = torch.min(q1_pi, q2_pi)  # Min of two Q-values

            # Calculate actor loss:    
            actor_loss = (self.alpha * pi_log_prob - min_q_pi).mean()

            # Optimize the actor network    
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.actor_loss = actor_loss.item()

            # Unfreeze critic gradients
            for p in self.critic.parameters():
                p.requires_grad = True


        if alpha_update:
            # Get actions and log probabilities for the current states from the actor
            if not a_update:
                pi_action, pi_log_prob = self.actor(state_batch)
            alpha_loss = -(self.log_alpha * (pi_log_prob.detach().float() + self.target_alpha)).mean() # type: ignore
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha_loss = alpha_loss.item()

        return self.actor_loss, self.critic_loss, self.alpha_loss, self.alpha
    
    def store_transition(self, state, action, reward, next_state, done, success):

        """
        Strores a transition to replay buffer. Values don't have to be torch tensors

        inputs:
            - state
            - action
            - reward
            - next_state
            - done:         Flag if last transition in episode
            - success       Flag if transition ended in target set
        """

        assert isinstance(reward, (float, np.ndarray, torch.Tensor))

        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32)
        done_tensor = torch.as_tensor(done, dtype=torch.float32)
        success_tensor = torch.as_tensor(success, dtype = torch.int)

        if isinstance(reward, float):
            transition = TensorDict({"s": state_tensor, "a": action_tensor, "r": reward_tensor, 
                                    "s'": next_state_tensor, "done": done_tensor, "success": success_tensor})
            
            self.replay_buffer.add(transition)

        elif isinstance(reward, np.ndarray):
            for s, a, r, s_next, d, suc in zip(state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor, success_tensor):
                transition = TensorDict({"s": s, "a": a, "r": r, "s'": s_next, "done": d, "success": suc})

                self.replay_buffer.add(transition)
    
    def get_action(self, state, deterministic = False):

        """
        Evaluates feedback law given by actor network for a given state. If deterministic returns mean of normal distribution, samples action from distribution otherwise.

        inputs:
            - state:            current state of agent to compute action for
            - deterministic:    whether to use mean of gaussian distribution or sample action from distribution

        outputs:
            - Action as ndarray
        """
        
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _ = self.actor(state_tensor, deterministic)

        return action.cpu().numpy()
    
    def store_weights(self, path):

        """
        Stores model weights to specified directory.

        inputs:
            - path: path to store NN weights at
        """

        torch.save(self.actor.state_dict(), path + "_actor.pt")
        torch.save(self.critic.state_dict(), path + "_critic.pt")

    def load_weights(self, path):
        
        """
        Loads NN weights from specified directory.

        inputs:
            - path: path to fetch weights from
        """

        actor_dict = torch.load(path + "_actor.pt", map_location=self.device)
        critic_dict = torch.load(path + "_critic.pt", map_location=self.device)

        self.actor.load_state_dict(actor_dict, strict=False)
        self.critic.load_state_dict(critic_dict)
        self.target_critic.load_state_dict(critic_dict)

    def store_model(self, path):
        
        """
        Stores actor and critic networks as torch models to specified directory.

        inputs:
            - path: path to store NNs at
        """

        torch.save(self.actor, path + "_actor_model.pt")
        torch.save(self.critic, path + "_critic_model.pt")

    def load_model(self, path):
        
        """
        Loads actor and critic networks as torch models from specified directory.

        inputs:
            - path: path to load NNs from
        """

        self.actor = torch.load(path + "_actor_model.pt", map_location=self.device, weights_only=False)
        self.critic = torch.load(path + "_critic_model.pt", map_location=self.device, weights_only=False)