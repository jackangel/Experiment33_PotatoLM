import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import time

# --- Hyperparameters ---
block_size = 4096
chunk_size = 256
batch_size = 8
max_epochs = 2000
learning_rate = 3e-4
eval_interval = 500
n_embed = 384
n_layer = 6
n_head = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'cpm_chunked_final_checkpoint.pth'

torch.manual_seed(1337)

# --- 1. Data Loading ---
try:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print("Error: 'input.txt' not found."); exit()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- 2. RoPE Implementation ---
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim, max_seq_len=block_size):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, device=inv_freq.device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, :, :])
        self.register_buffer("sin_cached", freqs.sin()[None, :, :])

    def forward(self, x):
        seq_len = x.shape[1]
        cos = self.cos_cached[:, :seq_len, ...]; sin = self.sin_cached[:, :seq_len, ...]
        x1 = x[..., 0::2]; x2 = x[..., 1::2]
        rotated_x = torch.cat(((-x2 * sin) + (x1 * cos), (x1 * sin) + (x2 * cos)), dim=-1)
        return rotated_x

# --- 3. FINAL STABILIZED "CHUNKED" HIERARCHICAL LAYER ---
class ChunkedMultiHeadCardPassingLayer(nn.Module):
    def __init__(self, num_heads, embedding_dim, local_chunk_size):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.chunk_size = local_chunk_size
        
        self.mark_net = nn.Linear(embedding_dim, embedding_dim)
        self.gate_net = nn.Linear(embedding_dim, embedding_dim)
        self.card_norm = nn.LayerNorm(self.head_size)
        self.carry_norm = nn.LayerNorm(self.head_size)

        self.head_output_net = nn.Sequential(
            nn.Linear(self.head_size * 2, self.head_size * 2),
            nn.GELU(),
            nn.Linear(self.head_size * 2, self.head_size)
        )
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.ln = nn.LayerNorm(embedding_dim)

        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        assert T % self.chunk_size == 0, "Sequence length must be divisible by chunk size"
        num_chunks = T // self.chunk_size
        H = self.num_heads
        D = self.head_size

        # --- 1. Initial Projections (Parallel across all tokens) ---
        potential_marks = self.mark_net(x)
        gates = torch.sigmoid(self.gate_net(x))
        
        # --- 2. Reshape for Chunked Processing ---
        # Reshape from (B, T, C) to (B, H, N, S, D)
        # B=Batch, H=Heads, N=Num Chunks, S=Chunk Size, D=Head Dim
        def reshape_for_chunks(tensor):
            return tensor.view(B, num_chunks, self.chunk_size, H, D).permute(0, 3, 1, 2, 4)

        x_heads = reshape_for_chunks(x)
        potential_marks = reshape_for_chunks(potential_marks)
        gates = reshape_for_chunks(gates)
        
        gated_marks = gates * potential_marks # Shape: (B, H, N, S, D)

        # --- THE PARALLEL SCAN REPLACEMENT ---
        
        # --- Step 1: Intra-Chunk Scan ---
        # Calculate cumulative sum *within* each chunk in parallel.
        # The cumsum is performed on the 'S' dimension (dim=3).
        local_cumulative_marks = torch.cumsum(gated_marks, dim=3) # Shape: (B, H, N, S, D)

        # Get the total sum of each chunk by taking the last element of the cumulative sum.
        # This will be used to calculate the carry between chunks.
        chunk_sums = local_cumulative_marks[:, :, :, -1, :].clone() # Shape: (B, H, N, D)
        
        # --- Step 2: Inter-Chunk Scan (Calculating the carry) ---
        # We need the carry *entering* each chunk. This is the cumulative sum of all *previous* chunk totals.
        # We can achieve this by doing a cumsum and then shifting the results right.
        initial_carry = torch.zeros(B, H, 1, D, device=x.device, dtype=x.dtype)
        
        # Calculate cumulative sum across the 'N' dimension (dim=2).
        carry_values_intermediate = torch.cumsum(chunk_sums, dim=2) # Shape: (B, H, N, D)
        
        # Concatenate initial zero carry and slice to get the correct carry for each chunk.
        # This is the "shift right" operation.
        carries_for_each_chunk = torch.cat([initial_carry, carry_values_intermediate[:, :, :-1, :]], dim=2) # Shape: (B, H, N, D)

        # Apply layer norm to the carries before they are used.
        # unsqueeze to add the 'S' dimension for broadcasting.
        normalized_carries = self.carry_norm(carries_for_each_chunk).unsqueeze(3) # Shape: (B, H, N, 1, D)
        
        # --- Step 3: Combine ---
        # Add the appropriate carry to every position within its corresponding chunk.
        # The `normalized_carries` tensor broadcasts across the 'S' dimension.
        marks_with_carry = local_cumulative_marks + normalized_carries # Shape: (B, H, N, S, D)

        # The card passed to a token is the cumulative sum of marks *before* it.
        # We achieve this by prepending an initial card and slicing off the last value.
        initial_card = normalized_carries # Use the chunk's carry as the initial card
        cards_passed_local = torch.cat([initial_card, marks_with_carry[:, :, :, :-1, :]], dim=3)
        cards_passed = self.card_norm(cards_passed_local) # Shape: (B, H, N, S, D)

        # --- 4. Final Head Output and Projection (Parallel) ---
        combined_head_info = torch.cat([x_heads, cards_passed], dim=-1)
        head_output = self.head_output_net(combined_head_info) # Shape: (B, H, N, S, D)

        # Reshape back to (B, T, C) for the next layer
        out = head_output.permute(0, 2, 3, 1, 4).contiguous().view(B, T, C)
        
        out = self.proj(out)
        out = self.ln(out)
        return x + out

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.rope = RotaryPositionalEmbeddings(n_embed)
        self.blocks = nn.ModuleList([ChunkedMultiHeadCardPassingLayer(
            num_heads=n_head, 
            embedding_dim=n_embed,
            local_chunk_size=chunk_size
        ) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx)
        x = tok_emb + self.rope(tok_emb)
        for block in self.blocks:
            # Only use checkpointing during training to save memory.
            # During evaluation (chat), run the forward pass normally.
            if self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=20):
        # self.eval() is called in the chat() function before this is used
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            seq_len = idx_cond.shape[1]
            pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
            if pad_len != 0:
                padding = torch.zeros((idx_cond.shape[0], pad_len), dtype=torch.long, device=device)
                idx_cond = torch.cat([idx_cond, padding], dim=1)
            
            # No need for autocast here as we are in no_grad mode, but it doesn't hurt
            # with torch.amp.autocast(device_type=device):
            logits, _ = self(idx_cond)

            logits = logits[:, seq_len - 1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next

# --- Training Function ---
def train():
    print(f"--- Starting Training (Chunked Hierarchical Version) ---")
    model = LanguageModel().to(device)
    model.train() # Set model to training mode
    print(f"Device: {device}, Context: {block_size}, Chunk Size: {chunk_size}, Heads: {n_head}, Params: ~{sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler()

    num_iterations = (len(data) // (block_size * batch_size)) * max_epochs
    print(f"Training for {num_iterations} effective iterations...")

    for iter_num in range(num_iterations):
        xb, yb = get_batch()

        with torch.amp.autocast(device_type=device):
            logits, loss = model(xb, yb)

        if torch.isnan(loss):
            print(f"Loss became NaN at iteration {iter_num+1}. Halting.")
            break

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if (iter_num + 1) % eval_interval == 0:
            print(f"\nIteration {iter_num+1}/{num_iterations} | Loss: {loss.item():.4f}")
            model.eval() # Switch to eval mode for generation
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            for token_tensor in model.generate(idx=context, max_new_tokens=100):
                char = decode(token_tensor[0].tolist())
                print(char, end='', flush=True)
            model.train() # Switch back to train mode
            print("\n-------------------------")
    
    if 'loss' in locals() and not torch.isnan(loss):
        print(f"Training complete. Saving checkpoint to {CHECKPOINT_PATH}")
        torch.save(model.state_dict(), CHECKPOINT_PATH)

# --- Chat Function ---
def chat():
    print(f"Checkpoint found at '{CHECKPOINT_PATH}'. Entering chat mode.")
    print("Type 'exit' or 'quit' to end the session.")
    
    # --- Model Setup ---
    model = LanguageModel().to(device)
    print("Loading model...")
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        model.eval() # Set the model to evaluation mode
        print(f"Model loaded successfully. Params: ~{sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return

    # --- Interactive Loop ---
    while True:
        prompt = input("\n> ")
        if prompt.lower() in ['exit', 'quit']:
            break
        if not prompt:
            continue

        context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        print(prompt, end='', flush=True) # Print the initial prompt
        
        try:
            # The generate function is a generator, so we iterate through it
            for token_tensor in model.generate(idx=context, max_new_tokens=500, temperature=0.8, top_k=50):
                char = decode(token_tensor[0].tolist())
                print(char, end='', flush=True)
        except KeyboardInterrupt:
            # Allow user to stop generation with Ctrl+C
            print("\nGeneration interrupted.")
            continue
        except Exception as e:
            print(f"\nAn error occurred during generation: {e}")
            continue
            
    print("\nExiting chat mode.")

# --- Main Execution Logic ---
if __name__ == '__main__':
    if os.path.exists(CHECKPOINT_PATH):
        chat()
    else:
        print(f"Checkpoint '{CHECKPOINT_PATH}' not found. Starting training...")
        train()
