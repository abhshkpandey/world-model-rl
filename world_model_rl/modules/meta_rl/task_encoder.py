import torch
import torch.nn as nn
import torch.optim as optim

class TaskEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[256, 128, 64], dropout_rate=0.2):
        super(TaskEncoder, self).__init__()
        
        # Encoder Network with weight tying for reduced redundancy
        self.shared_weight = nn.Parameter(torch.randn(input_dim, hidden_dims[0]))
        layers = []
        in_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            layers.append(nn.Linear(in_dim, h_dim, bias=False))  # Removing bias for efficiency
            layers.append(nn.ReLU())  # Activation function for non-linearity
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Using Group Normalization instead of Layer Normalization for smaller batch sizes
        self.norm = nn.GroupNorm(8, hidden_dims[-1])
        
        # Adaptive Dropout (Dropout rate can be updated dynamically)
        self.dropout = nn.Dropout(p=dropout_rate)  
        
        # Latent Space (Mean and Log-Variance for Reparameterization)
        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Initialize Weights
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        h = x @ self.shared_weight  # Weight tying in the first transformation
        h = self.encoder(h)  # Forward pass through deep encoder layers
        h = self.norm(h)  # Apply Group Normalization
        h = self.dropout(h)  # Apply Dropout for regularization
        
        mean = self.fc_mean(h)  # Compute mean for latent representation
        logvar = self.fc_logvar(h)  # Compute log variance for latent representation
        
        return mean, logvar
    
    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar) + 1e-6  # Prevent numerical instability
        eps = torch.randn_like(std)  # Sample from standard normal distribution
        return mean + eps * std  # Apply reparameterization trick
    
    def encode(self, x):
        mean, logvar = self.forward(x)  # Encode input into latent space
        z = self.reparameterize(mean, logvar)  # Sample from latent space
        return z, mean, logvar

# Example Usage
input_dim = 10  # Adjust based on the state/action space
latent_dim = 5   # Task representation dimension
batch_size = 32  # Configurable batch size
hidden_dims = [256, 128, 64]  # Increased depth for scalability
encoder = TaskEncoder(input_dim, latent_dim, hidden_dims, dropout_rate=0.2)

# Example Forward Pass
x_sample = torch.randn(batch_size, input_dim)  # Batch of configurable size
z, mean, logvar = encoder.encode(x_sample)
print("Latent Task Embeddings Shape:", z.shape)

# Apply dynamic quantization for lightweight deployment
encoder = torch.quantization.quantize_dynamic(encoder, {nn.Linear}, dtype=torch.qint8)
