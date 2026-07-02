import os
# Set UniMol weight directory (adjust to your local path)
os.environ["UNIMOL_WEIGHT_DIR"] = "/public/home/renpx/software/UniMol"
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from unimol_tools import UniMolRepr

# Suppress verbose logs from unimol_tools
logging.getLogger("unimol_tools").setLevel(logging.ERROR)



# ---------------------------- Configuration ----------------------------
SMILES_FILE = "Database_smiles.xlsx"
BATCH_SIZE = 32          # Molecules per batch
OUTPUT_NPY = "unimol_cls_repr.npy"
OUTPUT_FAILED_TXT = "unimol_failed_smiles.txt"

# ---------------------------- Load SMILES ----------------------------
print("Loading SMILES from Excel...")
df = pd.read_excel(SMILES_FILE)
smiles_list = df['Smiles'].tolist()
print(f"Total molecules: {len(smiles_list)}")

# ---------------------------- UniMol Inference with Failure Tracking ----------------------------
def compute_unimol_embeddings(smiles_list, batch_size=32):
    """Compute cls_repr for a list of SMILES, return embeddings and list of failed SMILES."""
    model = UniMolRepr(data_type="molecule", model_name="unimolv1", remove_hs=False)
    
    all_embeddings = []
    failed_smiles = []
    
    # Process in batches
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Processing batches"):
        batch = smiles_list[i:i+batch_size]
        try:
            result = model.get_repr(batch, return_atomic_reprs=False)
            batch_emb = np.array(result["cls_repr"])   # shape: (len(batch), 512)
            all_embeddings.append(batch_emb)
        except Exception as e:
            # If whole batch fails, record each SMILES in this batch as failed
            print(f"Batch starting at index {i} failed: {e}")
            failed_smiles.extend(batch)
            # Append zeros to keep alignment (optional, but we skip to avoid shape mismatch)
            # Instead, we will later handle missing embeddings via reindexing.
            # For simplicity, we append zeros with correct dimension.
            batch_fail_emb = np.zeros((len(batch), 512))
            all_embeddings.append(batch_fail_emb)
    
    if all_embeddings:
        embeddings = np.vstack(all_embeddings)
    else:
        embeddings = np.empty((0, 512))
    
    # Validate per-molecule success by checking if any row is all-zero? Not reliable.
    # Better: track explicitly. For now, we return both.
    return embeddings, failed_smiles

print("Computing UniMol embeddings...")
embeddings, failed_list = compute_unimol_embeddings(smiles_list, batch_size=BATCH_SIZE)

# ---------------------------- Post-process and Save ----------------------------
print(f"\nEmbeddings shape: {embeddings.shape}")   # (n_molecules, 512)
print(f"Number of molecules that failed completely (batch-level): {len(failed_list)}")

# Save embeddings
np.save(OUTPUT_NPY, embeddings)
print(f"Embeddings saved to {OUTPUT_NPY}")

# Save failed SMILES list
if failed_list:
    with open(OUTPUT_FAILED_TXT, 'w') as f:
        for smi in failed_list:
            f.write(smi + "\n")
    print(f"Failed SMILES saved to {OUTPUT_FAILED_TXT}")
else:
    print("All molecules processed successfully.")

# Optionally, print first few failed molecules if any
if failed_list:
    print("\nFirst 5 failed SMILES (if any):")
    for smi in failed_list[:5]:
        print(smi)

# Additional check: count non-zero embeddings (most should be non-zero)
non_zero_count = np.sum(np.linalg.norm(embeddings, axis=1) > 1e-6)
print(f"Molecules with non-zero embedding norm: {non_zero_count} / {len(embeddings)}")