import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from text_encoder import TaskEncoder  # PEARL-style Task Inference
from transformers import TransformerEncoder, TransformerEncoderLayer

# Define Transformer-based RL² Exploration Strategy
class RL2TransformerPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(RL2TransformerPolicy, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        return self.fc(x)

# Define Optimized Meta-Learner with Transformer-based Exploration
class MetaRL(nn.Module):
    def __init__(self, model, task_encoder, exploration_policy, lr_inner=0.01, num_inner_steps=1):
        super(MetaRL, self).__init__()
        self.model = model
        self.task_encoder = task_encoder
        self.exploration_policy = exploration_policy
        self.lr_inner = lr_inner
        self.num_inner_steps = num_inner_steps
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss for Stability
    
    def forward(self, x, task_context):
        task_embedding = self.task_encoder(task_context)  # PEARL Task Inference
        x = torch.cat([x, task_embedding], dim=-1)  # Append task context
        action = self.exploration_policy(x)  # RL² Exploration
        return self.model(x), action
    
    def adapt(self, loss, params):
        grads = grad(loss, params, create_graph=True)
        return [p - self.lr_inner * g for p, g in zip(params, grads)]
    
    def meta_train_step(self, task_batch):
        meta_loss = torch.tensor(0.0, requires_grad=True)
        for (x_train, y_train, x_val, y_val, task_context) in task_batch:
            params = list(self.model.parameters())
            for _ in range(self.num_inner_steps):
                pred, _ = self.forward(x_train, task_context)
                loss = self.loss_fn(pred, y_train)
                params = self.adapt(loss, params)
            
            val_pred, _ = self.forward(x_val, task_context)
            val_loss = self.loss_fn(val_pred, y_val)
            meta_loss = meta_loss + val_loss  # Accumulate loss to avoid premature backpropagation
        
        meta_loss.backward()
        return meta_loss.item()

# Example usage
if __name__ == "__main__":
    model = nn.Sequential(
        nn.Linear(10 + 16, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 1)  # Increased model capacity
    )
    
    task_encoder = TaskEncoder(input_dim=10, latent_dim=16)  # PEARL Task Encoder
    exploration_policy = RL2TransformerPolicy(input_dim=26, hidden_dim=64, output_dim=10, num_layers=3, dropout=0.1)  # Transformer-based RL²
    meta_rl = MetaRL(model, task_encoder, exploration_policy)
    optimizer = optim.AdamW(meta_rl.parameters(), lr=0.001, weight_decay=1e-4)  # AdamW optimizer for better stability
    
    # Simulated training loop with mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(5):
        task_batch = [(torch.randn(5, 10), torch.randn(5, 1), torch.randn(5, 10), torch.randn(5, 1), torch.randn(5, 10)) for _ in range(4)]
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = meta_rl.meta_train_step(task_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"Epoch {epoch+1}, Meta Loss: {loss:.4f}")
