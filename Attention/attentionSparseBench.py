import torch
import torch.nn.functional as F
import time
import torch.cuda as cuda

# Random Sparse Attention
class SparseAttention(torch.nn.Module):
    def __init__(self, d_model, sparsity=0.5):
        super(SparseAttention, self).__init__()
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        self.sparsity = sparsity

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        mask = (torch.rand_like(attn_scores) > self.sparsity).float()
        
        attn_scores = attn_scores * mask
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        output = torch.matmul(attn_weights, V)
        return output

# Block Sparse Attention
class BlockSparseAttention(torch.nn.Module):
    def __init__(self, d_model, block_size=32):
        super(BlockSparseAttention, self).__init__()
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        self.block_size = block_size

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_scores = torch.zeros_like(torch.matmul(Q, K.transpose(-2, -1))) / self.scale

        for i in range(0, x.size(1), self.block_size):
            for j in range(0, x.size(1), self.block_size):
                attn_scores[:, i:i+self.block_size, j:j+self.block_size] = torch.matmul(
                    Q[:, i:i+self.block_size], K[:, j:j+self.block_size].transpose(-2, -1)
                ) / self.scale

        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output

def benchmark_attention(model, x):
    start_time = time.time()
    output = model(x)
    end_time = time.time()

    time_taken = end_time - start_time
    memory_allocated = cuda.memory_allocated() / 1024**2 if cuda.is_available() else 0
    
    print(f"Time taken: {time_taken:.6f} seconds")
    print(f"Memory allocated: {memory_allocated:.2f} MB")
    return output

batch_size = 32
seq_length = 128
d_model = 64
sparsity = 0.7
block_size = 32

x = torch.randn(batch_size, seq_length, d_model)
#if cuda.is_available():
#    x = x.to('cuda')

sparse_attention = SparseAttention(d_model, sparsity=sparsity)
block_sparse_attention = BlockSparseAttention(d_model, block_size=block_size)

#if cuda.is_available():
#    sparse_attention = sparse_attention.to('cuda')
#    block_sparse_attention = block_sparse_attention.to('cuda')

print("Random Sparse Attention Benchmark:")
sparse_output = benchmark_attention(sparse_attention, x)

print("\nBlock Sparse Attention Benchmark:")
block_sparse_output = benchmark_attention(block_sparse_attention, x)

