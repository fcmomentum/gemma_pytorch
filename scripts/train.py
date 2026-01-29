import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import contextlib

# Add parent directory to path to import gemma modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemma import config
from gemma import model as gemma_model
from scripts.data import SplitBrainDataset

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

def train(args):
    # Model Config
    model_config = config.get_model_config(args.variant)
    model_config.quant = args.quant
    # For training we usually want float32 or bfloat16, but stick to float32 for simplicity unless specified
    dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
    model_config.dtype = "bfloat16" if dtype == torch.bfloat16 else "float32"

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # Load Model
    print(f"Loading model {args.variant}..." )
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma_model.GemmaForCausalLM(model_config)
        if args.ckpt:
            model.load_weights(args.ckpt)
        model = model.to(device)
    
    # Dataset
    dataset = SplitBrainDataset(
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        seq_len=args.seq_len,
        sliding_window_step=args.seq_len // 2, # 50% overlap default
        split="train",
        cache_dir=args.dataset_cache_dir,
        max_samples=args.max_samples
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    
    step = 0
    k = args.memory_lag # The 'k' for H_t vs H_{t-k}

    print("Starting training...")
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader)
        for batch in pbar:
            batch = batch.to(device)
            
            # Prepare inputs
            # Input: tokens[:-1], Target: tokens[1:]
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            
            batch_size, seq_len = input_ids.shape
            
            # Create dummy positions and masks
            input_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            # Standard causal mask
            mask = torch.full((1, 1, seq_len, seq_len), -2.3819763e38).to(torch.float)
            mask = torch.triu(mask, diagonal=1).to(device)
            # Create mask for each position
            mask = mask.expand(batch_size, -1, -1, -1) 
            # Note: Gemma forward expects mask to be selected per position usually in generation, 
            # but for bulk forward we need the full mask.
            # In GemmaModel.forward: scores = scores + mask
            # mask shape should be [batch, 1, seq_len, seq_len] broadcastable
            
            # Empty KV cache for training
            kv_caches = []
            for _ in range(model_config.num_hidden_layers):
                kv_caches.append((None, None)) # Using None to indicate no caching or letting model handle it for training if supported.
            # Actually, GemmaModel expects tuples of tensors for kv_cache.
            # But for training with full context we don't usually use recurrent KV cache.
            # However, the provided GemmaModel implementation is optimized for inference with KV cache.
            # We need to adapt it or construct valid dummy KV caches.
            # Re-reading GemmaModel.forward:
            # kv_cache: Tuple[torch.Tensor, torch.Tensor]
            # k_cache, v_cache = kv_cache
            # k_cache.index_copy_(1, kv_write_indices, xk)
            # It expects pre-allocated caches.
            
            # For training, we usually want to recompute everything.
            # We can allocate a cache of size seq_len for this batch.
            kv_caches = []
            for _ in range(model_config.num_hidden_layers):
                size = (batch_size, seq_len, model_config.num_key_value_heads, model_config.head_dim)
                k_c = torch.zeros(size, dtype=dtype, device=device)
                v_c = torch.zeros(size, dtype=dtype, device=device)
                kv_caches.append((k_c, v_c))

            # Forward pass
            # We need to pass output_positions for the sampler, but we want full logits.
            # The sampler in gemma/model.py selects indices: hidden_states = hidden_states.index_select(1, output_positions)
            # This implementation assumes we only want logits for specific positions (generation).
            # For training, we want logits for ALL positions.
            # Problem: The provided Sampler class is hardcoded to index_select.
            # Workaround: Pass all positions as output_positions?
            # output_positions: 1D tensor of indices?
            # Sampler: hidden_states.index_select(1, output_positions).squeeze(dim=1)
            # If we pass multiple positions, index_select will pick them, but squeeze(dim=1) might fail or squash if dim=1 is 1.
            # If output_positions has length N, result is [batch, N, hidden]. squeeze(1) -> [batch, hidden] ? No.
            # squeeze(dim=1) removes dimension 1. If dim 1 is size N > 1, it cannot be squeezed?
            # Actually, check Sampler code:
            # hidden_states = hidden_states.index_select(1, output_positions).squeeze(dim=1)
            # If output_positions is [0, 1, 2...], resulting shape is [batch, seq_len, hidden].
            # squeeze(dim=1) would fail if seq_len > 1.
            
            # So the existing Sampler is for generation (single step or specific tokens).
            # We need to bypass the sampler or modify it.
            # The cleanest way is to use the returned hidden_states and project them manually if needed, 
            # Or assume we can modify GemmaForCausalLM to return raw logits if output_positions is None?
            # Since I already modified GemmaForCausalLM, I can just compute logits manually from hidden_states.
            # Logits = hidden_states @ embedder.weight.T
            
            _, _, all_hidden_states = model(
                input_token_ids=input_ids,
                input_positions=input_positions,
                kv_write_indices=input_positions[0], # Write all positions, must be 1D for index_copy_
                kv_caches=kv_caches,
                mask=mask,
                output_positions=torch.tensor([seq_len-1], device=device), # Dummy
                temperatures=None,
                top_ps=torch.tensor([1.0], device=device),
                top_ks=torch.tensor([100], device=device),
                top_ks=torch.tensor([100], device=device),
                return_hidden_states=True
            )
            
            # DEBUG: Check if inputs or weights are zero
            if step == 0:
                 print(f"DEBUG: Input IDs sample: {input_ids[0, :10]}")
                 print(f"DEBUG: Embedder weight mean: {model.embedder.weight.mean().item()}")
                 print(f"DEBUG: All Hidden States 0 mean: {all_hidden_states[0].abs().mean().item()}")
                 print(f"DEBUG: All Hidden States -1 mean: {all_hidden_states[-1].abs().mean().item()}")
            
            # The last hidden state is the one used for prediction.
            # But wait, GemmaModel loop returns `hidden_states` which is the output of the last block.
            # Then it calls `self.norm(hidden_states)`.
            # `all_hidden_states` contains un-normalized outputs of layers?
            # My modification: 
            # hidden_states = self.norm(hidden_states)
            # if output_hidden_states: return hidden_states, all_hidden_states
            # So `hidden_states` is the final normalized state.
            
            final_hidden_state = all_hidden_states[-1] # This is technically pre-norm if I captured inside loop.
            # Wait, verify my capture logic.
            # loop: hidden_states = layer(...); append(hidden_states)
            # after loop: hidden_states = norm(hidden_states); return hidden_states, all
            # So all_hidden_states[-1] is the output of the last layer BEFORE final norm.
            # The `hidden_states` returned as first tuple element IS normalized.
            # We should use the first element for logits.
            
            # However, `model` returns `next_tokens, logits, all_hidden_states`.
            # `logits` from `sampler` only contains the last token (or whatever output_positions specified).
            # We need full sequence logits.
            
            last_hidden_state_normalized = all_hidden_states[-1] # Wait, no.
            # I need the final state returned by GemmaModel.forward, which is `hidden_states`.
            # In GemmaForCausalLM.forward, I unpack: `hidden_states, all_hidden_states = hidden_states`
            # And `hidden_states` (the normalized one) is passed to sampler.
            # But I don't return `hidden_states` (the normalized one) in the tuple `(next_tokens, logits, all_hidden_states)`.
            # I only return `all_hidden_states`. 
            # I should recall that `all_hidden_states` captured inside the loop are the layer outputs.
            # The final normalized state is NOT in `all_hidden_states`.
            
            # I should recalculate logits manually.
            # But I need the final normalized hidden state.
            # I should have returned it.
            
            # Let's fix GemmaForCausalLM to return it or just re-normalize.
            # `model.model.norm` is accessible.
            final_layer_output = all_hidden_states[-1]
            final_normalized = model.model.norm(final_layer_output)
            
            # Compute Logits
            embedder_weight = model.embedder.weight
            if model_config.quant:
                embedder_weight = embedder_weight * model.embedder.weight_scaler.unsqueeze(-1)
            
            logits = torch.matmul(final_normalized, embedder_weight.t())
            # Use float32 for loss
            logits = logits.float()
            
            # Cross Entropy Loss
            loss_xent = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Auxiliary Memory Loss
            if args.memory_weight > 0:
                # Get the intermediate layer
                layer_idx = args.memory_layer
                if layer_idx < 0: layer_idx += len(all_hidden_states)
                
                mem_hidden = all_hidden_states[layer_idx] # [Batch, Seq, Hidden]
                
                # Compare H_t with H_{t-k}
                # t starts from k.
                current = mem_hidden[:, k:, :]
                past = mem_hidden[:, :-k, :].detach() # Stop gradient on past
                
                loss_mem = F.l1_loss(current, past)
                # DEBUG PRINT
                if step % 10 == 0:
                     print(f"DEBUG: MemWeight={args.memory_weight}, Layer={layer_idx}, K={k}")
                     print(f"DEBUG: Current mean={current.mean().item():.6f}, Past mean={past.mean().item():.6f}")
                     print(f"DEBUG: Diff mean={torch.abs(current-past).mean().item():.6f}")
                     print(f"DEBUG: LossMem={loss_mem.item():.6f}")
            else:
                loss_mem = torch.tensor(0.0, device=device)

            total_loss = loss_xent + args.memory_weight * loss_mem

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Epoch {epoch} | Loss: {total_loss.item():.4f} (Xent: {loss_xent.item():.4f}, Mem: {loss_mem.item():.4f})")
            step += 1

    print("Training complete.")
    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to text directory or file")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer model")
    parser.add_argument("--variant", type=str, default="2b", help="Model variant")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--quant", action="store_true", help="Quantization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Memory Args
    parser.add_argument("--memory_weight", type=float, default=0.1, help="Weight of memory loss")
    parser.add_argument("--memory_layer", type=int, default=10, help="Layer index for memory loss")
    parser.add_argument("--memory_lag", type=int, default=1, help="Lag k for memory loss")
    parser.add_argument("--save_path", type=str, default="model_finetuned.pt", help="Save path")
    parser.add_argument("--dataset_cache_dir", type=str, default=".cache", help="Dataset cache dir")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")

    args = parser.parse_args()
    train(args)
