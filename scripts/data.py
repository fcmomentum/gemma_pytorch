import os
import pickle
import torch
from torch.utils.data import Dataset
from gemma import tokenizer
from tqdm import tqdm

class SplitBrainDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        seq_len: int = 2048,
        sliding_window_step: int = 512,
        cache_dir: str = ".cache",
        force_process: bool = False,
        split: str = "train"
    ):
        """
        Args:
            data_path: Path to the raw text file OR directory containing text files (e.g. PG19).
            tokenizer_path: Path to the sentencepiece model.
            seq_len: Length of sequences to output.
            sliding_window_step: Stride for sliding window.
            cache_dir: Directory to store pickle cache.
            force_process: If True, ignore cache and re-process.
            split: 'train', 'validation', or 'test' (used if data_path is a directory).
        """
        self.seq_len = seq_len
        self.sliding_window_step = sliding_window_step
        self.tokenizer = tokenizer.Tokenizer(tokenizer_path)
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Determine unique cache name based on path
        path_hash = str(abs(hash(data_path + split)))
        cache_path = os.path.join(cache_dir, f"dataset_{split}_{path_hash}.pkl")

        if os.path.exists(cache_path) and not force_process:
            print(f"Loading cached dataset from {cache_path}")
            with open(cache_path, "rb") as f:
                self.token_ids = pickle.load(f)
        else:
            print(f"Processing {data_path} for split '{split}'...")
            
            files_to_read = []
            if os.path.isdir(data_path):
                # Check for split subdirectory
                split_dir = os.path.join(data_path, split)
                if os.path.exists(split_dir):
                    search_dir = split_dir
                else:
                    search_dir = data_path
                
                # Find all .txt files
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith(".txt"):
                            files_to_read.append(os.path.join(root, file))
                files_to_read.sort() # Ensure deterministic order
            elif os.path.isfile(data_path):
                files_to_read = [data_path]
            else:
                 # Dummy fallback if path definitely doesn't exist (and not checked by caller)
                 print(f"Warning: {data_path} not found. Creating a dummy file.")
                 dummy_path = "dummy_input.txt"
                 with open(dummy_path, "w") as f:
                     f.write("This is a dummy text file for testing purposes. " * 1000)
                 files_to_read = [dummy_path]

            all_token_ids = []
            print(f"Found {len(files_to_read)} files.")
            
            # Process files
            for file_path in tqdm(files_to_read, desc="Tokenizing files"):
                 with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                     text = f.read()
                 # Encode and append EOS
                 tokens = self.tokenizer.encode(text, bos=True, eos=True)
                 all_token_ids.extend(tokens)
            
            self.token_ids = all_token_ids
            
            print(f"Saving dataset to {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(self.token_ids, f)

        # Calculate number of samples
        self.num_samples = max(0, (len(self.token_ids) - self.seq_len) // self.sliding_window_step + 1)
        print(f"Dataset loaded: {len(self.token_ids)} tokens, {self.num_samples} samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.sliding_window_step
        end_idx = start_idx + self.seq_len
        
        chunk = self.token_ids[start_idx:end_idx]
        
        # Determine targets (next token prediction)
        # Input: x_0, ..., x_{T-1}
        # Target: x_1, ..., x_T
        # But commonly in HF style we just pass input_ids and labels.
        # For Gemma custom loop, we might need manual shifting.
        # Let's return the raw chunk as input_ids.
        
        return torch.tensor(chunk, dtype=torch.long)
