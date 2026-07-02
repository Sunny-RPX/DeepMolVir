import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

# ========================== Configuration ==========================
CSV_FILE = "moses_dataset_v1.csv"          # Input CSV file
CHUNK_SIZE = 10000                         # Rows per CSV chunk
OUTPUT_NPY = "moses_ecfp4_2048.npy"        # Output fingerprint array
OUTPUT_FAILED_TXT = "moses_ecfp4_failed_smiles.txt"  # Failed SMILES
RADIUS = 2                                 # ECFP4 radius (2 * radius = 4)
N_BITS = 2048                              # Fingerprint length
DENSE_TYPE = np.uint8                      # Store as 0/1 bytes (saves memory vs float)

# ========================== Helper function ==========================
def smiles_to_ecfp4(smiles, radius=RADIUS, n_bits=N_BITS):
    """Convert a SMILES string to ECFP4 bit vector (list of 0/1 ints)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    # Convert to list of ints (0/1) for easy array building
    return list(fp)

# ========================== Main processing ==========================
def main():
    # Pre-count total chunks for progress bar
    total_chunks = sum(1 for _ in pd.read_csv(CSV_FILE, chunksize=CHUNK_SIZE))
    reader = pd.read_csv(CSV_FILE, chunksize=CHUNK_SIZE)

    all_fps = []          # Collect fingerprint chunks
    failed_records = []   # List of (SMILES, split) that failed

    pbar = tqdm(reader, total=total_chunks, desc="Processing chunks")
    for chunk in pbar:
        smiles_list = chunk['SMILES'].tolist()
        splits_list = chunk['SPLIT'].tolist() if 'SPLIT' in chunk.columns else None

        chunk_fps = []    # Fingerprints for current chunk

        for idx, smi in enumerate(smiles_list):
            fp = smiles_to_ecfp4(smi)
            if fp is not None:
                chunk_fps.append(fp)
            else:
                # Record failed SMILES with split info
                split_val = splits_list[idx] if splits_list is not None else None
                failed_records.append((smi, split_val))
                # Append a zero fingerprint to keep array alignment
                chunk_fps.append([0] * N_BITS)

        # Convert chunk to numpy array (dense) and add to list
        if chunk_fps:
            chunk_arr = np.array(chunk_fps, dtype=DENSE_TYPE)
            all_fps.append(chunk_arr)

        pbar.set_postfix({"chunk_shape": chunk_arr.shape if 'chunk_arr' in locals() else (0, N_BITS)})

    # Merge all chunks
    if all_fps:
        final_fps = np.vstack(all_fps)
    else:
        final_fps = np.empty((0, N_BITS), dtype=DENSE_TYPE)

    print(f"\nTotal fingerprints shape: {final_fps.shape}")
    print(f"Number of invalid SMILES: {len(failed_records)}")

    # Save fingerprint array
    np.save(OUTPUT_NPY, final_fps)
    print(f"Fingerprints saved to {OUTPUT_NPY}")

    # Save failed SMILES list
    if failed_records:
        with open(OUTPUT_FAILED_TXT, 'w') as f:
            for smi, split_val in failed_records:
                if split_val is not None:
                    f.write(f"{smi}\t{split_val}\n")
                else:
                    f.write(f"{smi}\n")
        print(f"Failed SMILES saved to {OUTPUT_FAILED_TXT}")
    else:
        print("All SMILES valid, no failures.")

if __name__ == "__main__":
    main()