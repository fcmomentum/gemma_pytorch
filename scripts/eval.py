import os
import sys
import torch
import torch.nn.functional as F
import argparse
import math
from tqdm import tqdm
import contextlib

# Add parent directory to path to import gemma modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemma import config
from gemma import model as gemma_model
from gemma import tokenizer

try:
    import lm_eval
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    from lm_eval.models.huggingface import HFLM
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False
    print("Warning: lm-evaluation-harness not installed. Only perplexity evaluation is available.")
    # Dummy LM class to avoid NameError
    class LM: pass

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

class GemmaWrapper(LM):
    def __init__(self, args, model, tokenizer):
        super().__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size_per_gpu = args.batch_size
        self.device = torch.device(args.device)

    def loglikelihood(self, requests):
        res = []
        for context, continuation in tqdm(requests, desc="Running loglikelihood"):
            # This is a naive implementation (one by one). Batching is harder.
            # Tokenize
            ctx_tokens = self.tokenizer.encode(context, bos=True, eos=False)
            cont_tokens = self.tokenizer.encode(continuation, bos=False, eos=False)
            
            full_tokens = ctx_tokens + cont_tokens
            input_ids = torch.tensor(full_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            
            # Forward
            with torch.no_grad():
                # We need logits for the continuation.
                # Gemma forward expects positions.
                seq_len = input_ids.size(1)
                input_positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
                mask = torch.full((1, 1, seq_len, seq_len), -2.3819763e38).to(torch.float)
                mask = torch.triu(mask, diagonal=1).to(self.device)
                mask = mask.expand(1, -1, -1, -1)
                
                kv_caches = []
                for _ in range(self.model.config.num_hidden_layers):
                    # For evaluation, standard non-recurrent is fine if efficient
                    kv_caches.append((torch.zeros(1, seq_len, self.model.config.num_key_value_heads, self.model.config.head_dim, device=self.device),
                                      torch.zeros(1, seq_len, self.model.config.num_key_value_heads, self.model.config.head_dim, device=self.device)))
                                      
                # Note: creating large zero caches is wasteful. 
                # Better to pass None or handle efficiently? 
                # GemmaModel expects pre-allocated cache for `index_copy_`.
                # So we must allocate correctly.
                
                _, _, all_hidden_states = self.model(
                    input_token_ids=input_ids,
                    input_positions=input_positions,
                    kv_write_indices=input_positions,
                    kv_caches=kv_caches,
                    mask=mask,
                    output_positions=torch.tensor([seq_len-1], device=self.device), # output doesn't matter, we want full states
                    temperatures=None,
                    top_ps=torch.tensor([1.0], device=self.device),
                    top_ks=torch.tensor([100], device=self.device),
                    return_hidden_states=True
                )
                
                # Get logits
                final_state = self.model.model.norm(all_hidden_states[-1])
                embedder_weight = self.model.embedder.weight
                if self.model.config.quant:
                     embedder_weight = embedder_weight * self.model.embedder.weight_scaler.unsqueeze(-1)
                
                logits = torch.matmul(final_state, embedder_weight.t())
                
                # Calculate logprobs of continuation
                # Logits at index i predict token i+1.
                # We want prediction of `cont_tokens`.
                # `ctx_tokens` has length L1. `cont_tokens` has length L2.
                # `full_tokens` has length L1+L2.
                # Target indices for continuation are from L1 to L1+L2-1.
                # Corresponding logits are at indices L1-1 to L1+L2-2.
                
                ctx_len = len(ctx_tokens)
                cont_len = len(cont_tokens)
                
                # logits slice: [0, L1-1 : L1+L2-1]
                relevant_logits = logits[0, ctx_len-1 : ctx_len+cont_len-1]
                relevant_logits = F.log_softmax(relevant_logits.float(), dim=-1)
                
                # Gather targets
                target_ids = torch.tensor(cont_tokens, device=self.device)
                
                greedy_tokens = torch.argmax(relevant_logits, dim=-1)
                is_greedy = (greedy_tokens == target_ids).all().item()
                
                log_prob = torch.gather(relevant_logits, -1, target_ids.unsqueeze(-1)).sum()
                
                res.append((log_prob.item(), is_greedy))
                
        return res

    def loglikelihood_rolling(self, requests):
        # Rolling Loglikelihood
        res = []
        for (string,) in tqdm(requests, desc="Running rolling loglikelihood"):
            tokens = self.tokenizer.encode(string, bos=True, eos=False)
            # Todo: chunking for sliding window if too long. Implemented naive for now.
            # Assuming short enough for context window.
            input_ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            seq_len = input_ids.size(1)
            
            with torch.no_grad():
                input_positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
                mask = torch.full((1, 1, seq_len, seq_len), -2.3819763e38).to(torch.float)
                mask = torch.triu(mask, diagonal=1).to(self.device).expand(1, -1, -1, -1)
                
                kv_caches = []
                for _ in range(self.model.config.num_hidden_layers):
                    kv_caches.append((torch.zeros(1, seq_len, self.model.config.num_key_value_heads, self.model.config.head_dim, device=self.device),
                                      torch.zeros(1, seq_len, self.model.config.num_key_value_heads, self.model.config.head_dim, device=self.device)))

                _, _, all_hidden_states = self.model(
                    input_token_ids=input_ids,
                    input_positions=input_positions,
                    kv_write_indices=input_positions,
                    kv_caches=kv_caches,
                    mask=mask,
                    output_positions=torch.tensor([seq_len-1], device=self.device),
                    temperatures=None,
                    top_ps=torch.tensor([1.0], device=self.device),
                    top_ks=torch.tensor([100], device=self.device),
                    return_hidden_states=True
                )
                
                final_state = self.model.model.norm(all_hidden_states[-1])
                embedder_weight = self.model.embedder.weight
                if self.model.config.quant:
                     embedder_weight = embedder_weight * self.model.embedder.weight_scaler.unsqueeze(-1)
                logits = torch.matmul(final_state, embedder_weight.t())
                
                log_probs = F.log_softmax(logits.float(), dim=-1)
                
                # Targets are tokens[1:]
                # Logits[i] predicts tokens[i+1]
                # Shift logits: logits[:, :-1] predicts tokens[1:]
                
                shifted_logits = log_probs[:, :-1, :]
                shifted_tokens = input_ids[:, 1:]
                
                token_log_probs = torch.gather(shifted_logits, -1, shifted_tokens.unsqueeze(-1)).squeeze(-1)
                sum_log_prob = token_log_probs.sum().item()
                
                res.append(sum_log_prob)
        return res

    def generate_until(self, requests):
        res = []
        for context, gen_kwargs in tqdm(requests, desc="Running generate_until"):
            until = gen_kwargs.get("until", [])
            output_len = gen_kwargs.get("max_gen_toks", 64)
            # Naive generation using model.generate (which we didn't update to handle 'until' well, but valid for now)
            generated = self.model.generate(
                prompts=context,
                device=self.device,
                output_len=output_len,
                temperature=gen_kwargs.get("temperature", 0.0), # 0.0 is effectively argmax but model.generate takes float. 1e-5?
                top_p=gen_kwargs.get("top_p", 1.0),
                top_k=gen_kwargs.get("top_k", 100)
            )
            # generated is full string including prompt (in some implementations) or just completion. 
            # gemma/model.py generate returns list of strings "results".
            # Check gemma/model.py generate:
            # results.append(self.tokenizer.decode(trimmed_output))
            # It seems to decode tokens[len(prompt):] so it returns ONLY completion.
            
            completion = generated
            # Handle 'until' stop sequences
            for stop in until:
                if stop in completion:
                    completion = completion.split(stop)[0]
            
            res.append(completion)
        return res

def eval_perplexity(args, model, tokenizer):
    print(f"Calculating perplexity on {args.ppl_file}...")
    with open(args.ppl_file, "r") as f:
        text = f.read()
    
    # Tokenize
    tokens = tokenizer.encode(text, bos=True, eos=True)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=args.device)
    
    seq_len = args.seq_len
    # Sliding window
    stride = seq_len
    nlls = []
    
    for i in tqdm(range(0, input_ids.size(0) - 1, stride)):
        start = i
        end = min(input_ids.size(0), start + seq_len)
        if end - start < 2: break
        
        chunk = input_ids[start:end].unsqueeze(0) # [1, L]
        
        with torch.no_grad():
            L = chunk.size(1)
            input_positions = torch.arange(L, device=args.device).unsqueeze(0)
            mask = torch.full((1, 1, L, L), -2.3819763e38).to(torch.float)
            mask = torch.triu(mask, diagonal=1).to(args.device).expand(1, -1, -1, -1)
            
            kv_caches = []
            for _ in range(model.config.num_hidden_layers):
                kv_caches.append((torch.zeros(1, L, model.config.num_key_value_heads, model.config.head_dim, device=args.device),
                                  torch.zeros(1, L, model.config.num_key_value_heads, model.config.head_dim, device=args.device)))
            
            _, _, all_hidden_states = model(
                input_token_ids=chunk,
                input_positions=input_positions,
                kv_write_indices=input_positions,
                kv_caches=kv_caches,
                mask=mask,
                output_positions=torch.tensor([L-1], device=args.device),
                temperatures=None,
                top_ps=torch.tensor([1.0], device=args.device),
                top_ks=torch.tensor([100], device=args.device),
                return_hidden_states=True
            )
            
            final_state = model.model.norm(all_hidden_states[-1])
            embedder_weight = model.embedder.weight
            if model.config.quant:
                 embedder_weight = embedder_weight * model.embedder.weight_scaler.unsqueeze(-1)
            logits = torch.matmul(final_state, embedder_weight.t())
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = chunk[..., 1:].contiguous()
            
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='sum')
            nlls.append(loss)
            
    ppl = torch.exp(torch.stack(nlls).sum() / (input_ids.size(0) - 1))
    print(f"Perplexity: {ppl.item():.4f}")
    return ppl.item()

def main(args):
    # Load Model
    model_config = config.get_model_config(args.variant)
    model_config.quant = args.quant
    dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
    model_config.dtype = "bfloat16" if dtype == torch.bfloat16 else "float32"

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"Loading model {args.variant}..." )
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma_model.GemmaForCausalLM(model_config)
        if args.ckpt:
            model.load_weights(args.ckpt)
        model = model.to(device)
        model.eval()

    tok = tokenizer.Tokenizer(args.tokenizer_path)

    if args.ppl_file:
        eval_perplexity(args, model, tok)

    if args.tasks and LM_EVAL_AVAILABLE:
        print(f"Running lm-eval on tasks: {args.tasks}")
        wrapper = GemmaWrapper(args, model, tok)
        results = lm_eval.simple_evaluate(
            model=wrapper,
            tasks=args.tasks.split(","),
            batch_size=args.batch_size,
        )
        print(results)
    elif args.tasks:
        print("Skipping lm-eval tasks because lm-evaluation-harness is not installed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--variant", type=str, default="2b", help="Model variant")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ppl_file", type=str, default=None, help="Text file for perplexity")
    parser.add_argument("--tasks", type=str, default=None, help="Comma separated lm-eval tasks")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--quant", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)
