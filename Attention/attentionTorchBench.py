import torch
import torch.nn.functional as F
import time

# Standard Attention Implementation
class StandardAttention(torch.nn.Module):
    def __init__(self, d_model):
        super(StandardAttention, self).__init__()
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Weighted sum of values
        output = torch.matmul(attn_weights, V)
        return output

# Fast Attention (Approximated using Performer)
class FastAttention(torch.nn.Module):
    def __init__(self, d_model, kernel_transformation=torch.nn.ReLU):
        super(FastAttention, self).__init__()
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.kernel_transformation = kernel_transformation()

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Apply kernel transformation
        Q = self.kernel_transformation(Q)
        K = self.kernel_transformation(K)

        # Compute attention approximation
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        output = torch.matmul(attn_weights, V)
        return output


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

batch_size = 32
seq_length = 128
d_model = 64

# Input Tensor
x = torch.randn(batch_size, seq_length, d_model).to(device)

standard_attention = StandardAttention(d_model).to(device)
fast_attention = FastAttention(d_model).to(device)

# Standard Attention
start_time = time.time()
standard_output = standard_attention(x)
standard_time = time.time() - start_time
print(f"Standard Attention Time: {standard_time:.6f} seconds")

# Fast Attention
start_time = time.time()
fast_output = fast_attention(x)
fast_time = time.time() - start_time
print(f"Fast Attention Time: {fast_time:.6f} seconds")

