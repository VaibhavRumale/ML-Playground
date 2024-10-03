import jax
import jax.numpy as jnp
import time

# Standard Attention in JAX
def standard_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = jnp.matmul(Q, K.transpose((0, 2, 1))) / jnp.sqrt(d_k)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(attn_weights, V)
    return output

# approximation using kernel transformations
def fast_attention(Q, K, V):
    Q = jax.nn.relu(Q)
    K = jax.nn.relu(K)
    scores = jnp.matmul(Q, K.transpose((0, 2, 1)))
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(attn_weights, V)
    return output

batch_size = 32
seq_length = 128
d_model = 64


key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (batch_size, seq_length, d_model))

def linear_transform(x, d_model):
    w = jax.random.normal(key, (d_model, d_model))
    return jnp.dot(x, w)

Q = linear_transform(x, d_model)
K = linear_transform(x, d_model)
V = linear_transform(x, d_model)

# Standard Attention
start_time = time.time()
standard_output = standard_attention(Q, K, V)
standard_time = time.time() - start_time
print(f"Standard Attention Time: {standard_time:.6f} seconds")

# Fast Attention
start_time = time.time()
fast_output = fast_attention(Q, K, V)
fast_time = time.time() - start_time
print(f"Fast Attention Time: {fast_time:.6f} seconds")

