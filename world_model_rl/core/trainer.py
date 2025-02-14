import torch
import torch.optim as optim
from core.config import CONFIG

class Trainer:
    def __init__(self, model, config=CONFIG):
        self.model = model.to(config["device"])
        self.config = config
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        self.criterion = torch.nn.MSELoss()
    
    def train(self):
        for epoch in range(self.config["num_epochs"]):
            # Simulated training loop
            inputs = torch.randn(32, 10).to(self.config["device"])
            targets = torch.randn(32, 1).to(self.config["device"])
            
            self.optimizer.zero_grad()
            outputs, _ = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            print(f"Epoch {epoch+1}/{self.config['num_epochs']}, Loss: {loss.item():.4f}")
