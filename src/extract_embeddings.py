"""
Extract embeddings from protein sequences using ESM2-650M model
"""

from transformers import EsmModel, EsmTokenizer
import torch
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm

def load_esm2_model(model_path: str, device: str = "cuda") -> tuple[EsmModel, EsmTokenizer, str]:
    """
    Load the ESM2-650M model and  tokenizer from a given path.

    Args:
        model_path: Path to the ESM2-650M model
        device: Device to use for the model
    
    Returns:
        EsmModel: The ESM2-650M model
        EsmTokenizer: The ESM2-650M tokenizer
        device: Device used for the model
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    model = EsmModel.from_pretrained(model_path)
    # tokenizer does not need to be moved
    tokenizer = EsmTokenizer.from_pretrained(model_path)

    model = model.to(device)
    model.eval()

    print("ESM2-650M Model Loaded Successfully on", device)

    return model, tokenizer, device

def extract_embeddings(sequences: list[str], model: EsmModel, tokenizer: EsmTokenizer, device: str = "cuda", batch_size: int = 32) -> np.ndarray:
    """
    Tokenizes sequences in batches.

    Args:
        sequences: List of protein sequences
        model: ESM2-650M Model
        tokenizer: ESM2-650M Tokenizer
        device: Device to use for the model
        batch_size: Batch size for the model

    Returns:
        np.ndarray: Array of embeddings of shape [num_sequences, hidden_size] with mean_pooled embeddings
    """

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_sequences = sequences[i:i+batch_size]
        
        encoded = tokenizer(
            batch_sequences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        hidden_states = outputs.last_hidden_state

        sum_embeddings = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1)
        sum_mask = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_pooled = sum_embeddings / sum_mask

        all_embeddings.append(mean_pooled.cpu().numpy())

    return np.vstack(all_embeddings)