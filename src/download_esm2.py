from transformers import EsmModel, EsmTokenizer

from typing import Optional

import argparse

"""
Download the ESM2-650M protein language model and tokenizer from Hugging Face and saves it to the given path if given.
"""

def download_esm2_650m(model_path: Optional[str] = None, cache_dir: Optional[str] = None) -> tuple[EsmModel, EsmTokenizer]:

    """
    Download the ESM2-650M protein language model and tokenizer from Hugging Face and saves it to the given path if given.

    Args:
    - model_path: Path to save the model and tokenizer
    - cache_dir: Path to the cached directory

    Returns:
    - model: ESM2-650M protein language model
    - tokenizer: ESM2-650M protein language tokenizer
    """

    identifier = "facebook/esm2_t33_650M_UR50D"

    try:
        print(f"Downloading ESM2-650M Model: {identifier}")
        model = EsmModel.from_pretrained(identifier, cache_dir=cache_dir)
        print("ESM2-650M Model Downloaded Successfully")

        print(f"Downloading ESM2-650M Tokenizer: {identifier}")
        tokenizer = EsmTokenizer.from_pretrained(identifier, cache_dir=cache_dir)
        print("ESM2-650M Tokenizer Downloaded Successfully")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

    if model_path:        
        print(f"Saving ESM2-650M Model to: {model_path}")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print("ESM2-650M Model and Tokenizer Saved Successfully")

    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Download the ESM2-650M protein language model and tokenizer from Hugging Face and saves it to the given path if given."
    )

    parser.add_argument("--model_path", type=str, required=False, help="Path to save the model and tokenizer")
    parser.add_argument("--cache_dir", type=str, required=False, help="Path to the cached directory")

    args = parser.parse_args()

    download_esm2_650m(args.model_path, args.cache_dir)

if __name__ == "__main__":
    main()