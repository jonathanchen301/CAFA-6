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