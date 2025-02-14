import torch
import torch.optim as optim
from core.config import CONFIG
from core.trainer import Trainer
from world_model_rl import WorldModelRL

if __name__ == "__main__":
    model = WorldModelRL()
    trainer = Trainer(model, CONFIG)
    trainer.train()
