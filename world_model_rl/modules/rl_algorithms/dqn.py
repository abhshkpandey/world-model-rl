import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
    
    def select_action(self, state):
        with torch.no_grad():
            return self.model(state).argmax().item()
    
    def train_step(self, state, action, reward, next_state, done):
        q_values = self.model(state)
        next_q_values = self.model(next_state).max(1)[0].detach()
        target_q_value = reward + (1 - done) * self.gamma * next_q_values
        loss = F.mse_loss(q_values.gather(1, action.unsqueeze(1)).squeeze(), target_q_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
