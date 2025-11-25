import pandas as pd
from Bio import SeqIO

def fasta2df(path: str) -> pd.DataFrame:
    """
    Convert a FASTA file to a DataFrame.
    
    Args:
        path: Path to the FASTA file.
    
    Returns:
        DataFrame with columns "id" and "sequence".
    """
    def extract_protein_id(rec_id: str) -> str:
        """Extract the actual protein ID from different FASTA header formats."""
        # Handle UniProt format: sp|PROTEIN_ID|NAME -> extract PROTEIN_ID
        if '|' in rec_id:
            parts = rec_id.split('|')
            if len(parts) >= 2:
                return parts[1]  # Extract the middle part (protein ID)
        # Handle simple format: PROTEIN_ID taxon_id -> extract PROTEIN_ID
        return rec_id.split()[0]  # Take first token before space
    
    records = []
    for rec in SeqIO.parse(path, "fasta"):
        protein_id = extract_protein_id(rec.id)
        records.append((protein_id, str(rec.seq)))

    return pd.DataFrame(records, columns=["id", "sequence"])