import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TRPO(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(TRPO, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

class TRPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99):
        self.model = TRPO(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
    
    def select_action(self, state):
        action_probs, _ = self.model(state)
        action = torch.multinomial(action_probs, 1)
        return action.item()
    
    def train_step(self, states, actions, rewards, next_states, dones):
        advantages = rewards + self.gamma * self.model.critic(next_states).detach() - self.model.critic(states)
        policy_loss = -torch.mean(torch.log(self.model.actor(states).gather(1, actions.unsqueeze(1)).squeeze()) * advantages)
        value_loss = F.mse_loss(self.model.critic(states), rewards + self.gamma * self.model.critic(next_states).detach())
        
        loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
