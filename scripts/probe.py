import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
from tqdm import tqdm
import contextlib

# Add parent directory to path to import gemma modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemma import config
from gemma import model as gemma_model
from gemma import tokenizer

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

def generate_nihs_data(tokenizer, num_samples, seq_len, num_classes=4, needle_depth_ratio=0.5):
    """
    Generates synthetic data.
    Classes: 0, 1, 2, 3 (corresponding to 4 different 'needles')
    """
    # Define 4 needles
    needles = [
        " The secret code is Alpha. ",
        " The secret code is Beta. ",
        " The secret code is Gamma. ",
        " The secret code is Delta. "
    ]
    
    data = []
    labels = []
    
    # Filler text
    filler = "The quick brown fox jumps over the lazy dog. " * 100
    filler_tokens = tokenizer.encode(filler)
    
    for _ in range(num_samples):
        label = random.randint(0, num_classes - 1)
        needle_str = needles[label]
        needle_tokens = tokenizer.encode(needle_str)
        
        # Construct sequence
        # We want total length ~ seq_len
        # Needle position depends on depth ratio
        
        target_len = seq_len
        
        # Calculate positions
        needle_pos = int(target_len * needle_depth_ratio)
        
        # Fill before
        tokens = []
        while len(tokens) < needle_pos:
            tokens.extend(filler_tokens)
        tokens = tokens[:needle_pos]
        
        # Add needle
        tokens.extend(needle_tokens)
        
        # Fill after
        while len(tokens) < target_len:
            tokens.extend(filler_tokens)
        tokens = tokens[:target_len]
        
        data.append(torch.tensor(tokens, dtype=torch.long))
        labels.append(label)
        
    return torch.stack(data), torch.tensor(labels, dtype=torch.long)

def probe(args):
    # Load Model
    model_config = config.get_model_config(args.variant)
    model_config.quant = args.quant
    dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
    model_config.dtype = "bfloat16" if dtype == torch.bfloat16 else "float32"

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"Loading model {args.variant}..." )
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma_model.GemmaForCausalLM(model_config)
        if args.ckpt:
            model.load_weights(args.ckpt)
        model = model.to(device)
        model.eval()

    # Create Tokenizer
    tok = tokenizer.Tokenizer(args.tokenizer_path)

    # Generate Data
    print("Generating NIHS data...")
    train_x, train_y = generate_nihs_data(tok, args.train_samples, args.seq_len, needle_depth_ratio=0.5)
    test_x, test_y = generate_nihs_data(tok, args.test_samples, args.seq_len, needle_depth_ratio=0.5)
    
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    
    # Extract Embeddings
    # We want to train the probe on the hidden state of the LAST token? 
    # Or the token where the needle was? 
    # Usually "retrieval" implies we can answer at the end.
    # So we take the hidden state of the last token.
    
    def get_embeddings(token_ids):
        batch_size, seq_len = token_ids.shape
        input_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = torch.full((1, 1, seq_len, seq_len), -2.3819763e38).to(torch.float)
        mask = torch.triu(mask, diagonal=1).to(device)
        mask = mask.expand(batch_size, -1, -1, -1) 
        
        kv_caches = []
        for _ in range(model_config.num_hidden_layers):
             kv_caches.append((None, None))
        
        # Run forward
        with torch.no_grad():
             _, _, all_hidden_states = model(
                input_token_ids=token_ids,
                input_positions=input_positions,
                kv_write_indices=input_positions,
                kv_caches=kv_caches,
                mask=mask,
                output_positions=torch.tensor([seq_len-1], device=device),
                temperatures=None,
                top_ps=torch.tensor([1.0], device=device),
                top_ks=torch.tensor([100], device=device),
                return_hidden_states=True
            )
        
        # Get target layer
        layer_idx = args.layer
        if layer_idx < 0: layer_idx += len(all_hidden_states)
        hidden = all_hidden_states[layer_idx] # [Batch, Seq, H]
        
        # Take last token
        last_hidden = hidden[:, -1, :].float() # Cast to float for probe training
        return last_hidden

    print("Extracting features...")
    # Batch processing if needed, but for small probe scale simple is fine
    train_feats = []
    batch_size = args.batch_size
    for i in range(0, len(train_x), batch_size):
        batch = train_x[i:i+batch_size]
        train_feats.append(get_embeddings(batch))
    train_feats = torch.cat(train_feats, 0)
    
    test_feats = []
    for i in range(0, len(test_x), batch_size):
        batch = test_x[i:i+batch_size]
        test_feats.append(get_embeddings(batch))
    test_feats = torch.cat(test_feats, 0)
    
    # Train Probe
    print(f"Training probe on layer {args.layer} features...")
    input_dim = train_feats.shape[1]
    probe_model = LinearProbe(input_dim, 4).to(device)
    optimizer = optim.Adam(probe_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.probe_epochs):
        optimizer.zero_grad()
        logits = probe_model(train_feats)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    # Evaluate
    print("Evaluating...")
    with torch.no_grad():
        logits = probe_model(test_feats)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == test_y).float().mean()
        
    print(f"Probe Accuracy at Layer {args.layer}: {acc.item():.4f}")
    return acc.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--variant", type=str, default="2b", help="Model variant")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layer", type=int, default=10, help="Layer to probe")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--train_samples", type=int, default=100)
    parser.add_argument("--test_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--probe_epochs", type=int, default=100)
    parser.add_argument("--quant", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    probe(args)
