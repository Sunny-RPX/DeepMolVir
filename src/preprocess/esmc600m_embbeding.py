# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

# Import ESMC-600M related modules (replace original ESM3)
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESMC_600M

# Enable offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# ESMC-600M embedding dimension
FIXED_EMBEDDING_DIM = 1152
skip_reasons = defaultdict(int)

# ---------------------------- Embedding Extraction Function ----------------------------
def get_esmc600_embedding(model_client, sequence, max_chunk_length=1000):
    """
    Calculate residue-averaged protein embedding with sequence chunking.
    Returns 1D numpy array with shape (1152,) if successful; returns None on failure.
    """
    try:
        embedding_sum = None
        total_residue_count = 0
        # Split long sequence into fixed-length chunks
        for start_idx in range(0, len(sequence), max_chunk_length):
            seq_chunk = sequence[start_idx:start_idx + max_chunk_length]
            protein_obj = ESMProtein(sequence=seq_chunk)
            protein_tensor = model_client.encode(protein_obj)
            model_output = model_client.forward_and_sample(
                protein_tensor,
                SamplingConfig(return_per_residue_embeddings=True)
            )
            residue_embeddings = model_output.per_residue_embedding  # Shape: (L, 1152)

            if embedding_sum is None:
                embedding_sum = residue_embeddings.sum(dim=0)
            else:
                embedding_sum += residue_embeddings.sum(dim=0)
            total_residue_count += residue_embeddings.shape[0]

        if total_residue_count == 0:
            return None
        average_embedding = embedding_sum / total_residue_count
        return average_embedding.cpu().numpy()
    except Exception:
        return None

# ---------------------------- Model Initialization ----------------------------
model_client = ESMC.from_pretrained(ESMC_600M)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_client = model_client.to(device)
print(f"ESMC-600M model loaded onto {device}")

# ---------------------------- Load Input Spreadsheet ----------------------------
df = pd.read_excel("Database_viruses.xlsx")
print(f"Total data rows loaded: {len(df)}")

# Column index definitions (match Excel sheet structure)
membrane_protein_cols = df.columns[15:19]      # 4 membrane protein columns
nuclear_protein_cols = df.columns[19:24]       # 5 nuclear protein columns
functional_protein_cols = df.columns[24:30]    # 6 functional protein columns
classification_protein_col = df.columns[30]    # 1 classification protein column

MAX_MEMBRANE_PROTEINS = 4
MAX_NUCLEAR_PROTEINS = 5
MAX_FUNCTIONAL_PROTEINS = 6
MAX_CLASSIFICATION_PROTEINS = 1

sample_storage = []  # Store embedding dictionaries for all valid samples

# ---------------------------- Iterate Over All Samples ----------------------------
for row_index, row_data in df.iterrows():
    row_number = row_index + 1
    virus_name = str(row_data.iloc[0]).strip() if pd.notna(row_data.iloc[0]) else f"Unnamed_Sample_{row_number}"

    # Skip sample if the first membrane protein entry is marked "None"
    first_membrane_entry = row_data.iloc[15]
    if isinstance(first_membrane_entry, str) and first_membrane_entry.strip() == "None":
        skip_reasons["first_membrane_protein_is_none"] += 1
        sample_storage.append(None)
        print(f"SKIP | Row {row_number} | Virus: {virus_name}")
        continue

    valid_sequence_counter = 0

    # Process membrane protein embeddings
    membrane_embedding_list = []
    for col in membrane_protein_cols:
        seq_text = row_data[col]
        if not isinstance(seq_text, str) or len(seq_text.strip()) == 0:
            membrane_embedding_list.append(np.zeros(FIXED_EMBEDDING_DIM))
            continue
        emb_result = get_esmc600_embedding(model_client, seq_text)
        if emb_result is not None:
            membrane_embedding_list.append(emb_result)
            valid_sequence_counter += 1
        else:
            membrane_embedding_list.append(np.zeros(FIXED_EMBEDDING_DIM))
    membrane_embedding_list = membrane_embedding_list[:MAX_MEMBRANE_PROTEINS]
    while len(membrane_embedding_list) < MAX_MEMBRANE_PROTEINS:
        membrane_embedding_list.append(np.zeros(FIXED_EMBEDDING_DIM))

    # Process nuclear protein embeddings
    nuclear_embedding_list = []
    for col in nuclear_protein_cols:
        seq_text = row_data[col]
        if not isinstance(seq_text, str) or len(seq_text.strip()) == 0:
            nuclear_embedding_list.append(np.zeros(FIXED_EMBEDDING_DIM))
            continue
        emb_result = get_esmc600_embedding(model_client, seq_text)
        if emb_result is not None:
            nuclear_embedding_list.append(emb_result)
            valid_sequence_counter += 1
        else:
            nuclear_embedding_list.append(np.zeros(FIXED_EMBEDDING_DIM))
    nuclear_embedding_list = nuclear_embedding_list[:MAX_NUCLEAR_PROTEINS]
    while len(nuclear_embedding_list) < MAX_NUCLEAR_PROTEINS:
        nuclear_embedding_list.append(np.zeros(FIXED_EMBEDDING_DIM))

    # Process functional protein embeddings
    functional_embedding_list = []
    for col in functional_protein_cols:
        seq_text = row_data[col]
        if not isinstance(seq_text, str) or len(seq_text.strip()) == 0:
            functional_embedding_list.append(np.zeros(FIXED_EMBEDDING_DIM))
            continue
        emb_result = get_esmc600_embedding(model_client, seq_text)
        if emb_result is not None:
            functional_embedding_list.append(emb_result)
            valid_sequence_counter += 1
        else:
            functional_embedding_list.append(np.zeros(FIXED_EMBEDDING_DIM))
    functional_embedding_list = functional_embedding_list[:MAX_FUNCTIONAL_PROTEINS]
    while len(functional_embedding_list) < MAX_FUNCTIONAL_PROTEINS:
        functional_embedding_list.append(np.zeros(FIXED_EMBEDDING_DIM))

    # Process classification protein embedding
    classification_embedding_list = []
    seq_text = row_data[classification_protein_col]
    if not isinstance(seq_text, str) or len(seq_text.strip()) == 0:
        classification_embedding_list.append(np.zeros(FIXED_EMBEDDING_DIM))
    else:
        emb_result = get_esmc600_embedding(model_client, seq_text)
        if emb_result is not None:
            classification_embedding_list.append(emb_result)
            valid_sequence_counter += 1
        else:
            classification_embedding_list.append(np.zeros(FIXED_EMBEDDING_DIM))
    classification_embedding_list = classification_embedding_list[:MAX_CLASSIFICATION_PROTEINS]
    while len(classification_embedding_list) < MAX_CLASSIFICATION_PROTEINS:
        classification_embedding_list.append(np.zeros(FIXED_EMBEDDING_DIM))

    # Skip sample when zero valid protein sequences exist
    if valid_sequence_counter == 0:
        skip_reasons["no_valid_protein_sequences"] += 1
        sample_storage.append(None)
        print(f"SKIP | Row {row_number} | Virus: {virus_name}")
        continue

    try:
        sample_dict = {
            "virus_name": virus_name,
            "membrane_embeds": np.stack(membrane_embedding_list),      # Shape: (4, 1152)
            "nuclear_embeds": np.stack(nuclear_embedding_list),        # Shape: (5, 1152)
            "functional_embeds": np.stack(functional_embedding_list),   # Shape: (6, 1152)
            "classification_embeds": np.stack(classification_embedding_list)  # Shape: (1, 1152)
        }
        sample_storage.append(sample_dict)
        print(f"OK | Row {row_number} | Virus: {virus_name}")
    except Exception:
        skip_reasons["embedding_shape_mismatch_error"] += 1
        sample_storage.append(None)

# ---------------------------- Generate Average Embeddings for Unclassified Virus Groups ----------------------------
# Filter samples with successfully computed embeddings
valid_samples = [item for item in sample_storage if item is not None]
valid_virus_names = np.array([item["virus_name"] for item in valid_samples])
valid_membrane_arrays = np.stack([item["membrane_embeds"] for item in valid_samples])       # (N_valid, 4, 1152)
valid_nuclear_arrays = np.stack([item["nuclear_embeds"] for item in valid_samples])           # (N_valid, 5, 1152)
valid_functional_arrays = np.stack([item["functional_embeds"] for item in valid_samples])     # (N_valid, 6, 1152)
valid_classification_arrays = np.stack([item["classification_embeds"] for item in valid_samples])  # (N_valid, 1, 1152)

# Mapping of unclassified virus labels to their corresponding genus prefixes for mean embedding filling
unclassified_virus_mapping = {
    "Echovirus-unclass": "Echovirus",
    "Enterovirus-unclass": "Enterovirus",
    "Poliovirus-unclass": "Poliovirus",
    "Rhinovirus-unclass": "Rhinovirus",
    "Human immunodeficiency virus-unclass": "Human immunodeficiency virus",
    "Human adenovirus-unclass": "Human adenovirus",
    "Influenza-unclass": "Influenza",
    "Dengue virus-unclass": "Dengue virus",
    "Hepatitis c virus genotype-unclass": "Hepatitis c virus genotype",
    "Human papillomavirus-unclass": "Human papillomavirus"
}

group_average_embeds = {}
for unclass_label, genus_prefix in unclassified_virus_mapping.items():
    virus_mask = np.array([name.startswith(genus_prefix) for name in valid_virus_names])
    if not virus_mask.any():
        print(f"WARNING: No matching samples found for genus prefix: {genus_prefix}")
        continue
    mean_membrane = valid_membrane_arrays[virus_mask].mean(axis=0)
    mean_nuclear = valid_nuclear_arrays[virus_mask].mean(axis=0)
    mean_functional = valid_functional_arrays[virus_mask].mean(axis=0)
    mean_classification = valid_classification_arrays[virus_mask].mean(axis=0)
    group_average_embeds[unclass_label] = {
        "membrane": mean_membrane,
        "nuclear": mean_nuclear,
        "functional": mean_functional,
        "classification": mean_classification
    }
    print(f"GROUP AVG | {unclass_label} | Matching sample count: {virus_mask.sum()}")

# ---------------------------- Construct Final Full Dataset in Original Row Order ----------------------------
final_virus_name_list = []
final_membrane_stack = []
final_nuclear_stack = []
final_functional_stack = []
final_classification_stack = []

for idx, sample_data in enumerate(sample_storage):
    original_virus_label = str(df.iloc[idx, 0]).strip()
    # Fill precomputed group average embedding for unclassified virus entries
    if original_virus_label in group_average_embeds:
        avg_embed_set = group_average_embeds[original_virus_label]
        final_virus_name_list.append(original_virus_label)
        final_membrane_stack.append(avg_embed_set["membrane"])
        final_nuclear_stack.append(avg_embed_set["nuclear"])
        final_functional_stack.append(avg_embed_set["functional"])
        final_classification_stack.append(avg_embed_set["classification"])
        print(f"FILL GROUP AVG | Virus: {original_virus_label}")
        continue

    # Append valid precomputed embedding
    if sample_data is not None:
        final_virus_name_list.append(sample_data["virus_name"])
        final_membrane_stack.append(sample_data["membrane_embeds"])
        final_nuclear_stack.append(sample_data["nuclear_embeds"])
        final_functional_stack.append(sample_data["functional_embeds"])
        final_classification_stack.append(sample_data["classification_embeds"])
    # Fill zero padding array for failed/skipped samples
    else:
        fallback_name = original_virus_label if pd.notna(df.iloc[idx, 0]) else f"Unnamed_Sample_{idx + 1}"
        final_virus_name_list.append(fallback_name)
        final_membrane_stack.append(np.zeros((MAX_MEMBRANE_PROTEINS, FIXED_EMBEDDING_DIM)))
        final_nuclear_stack.append(np.zeros((MAX_NUCLEAR_PROTEINS, FIXED_EMBEDDING_DIM)))
        final_functional_stack.append(np.zeros((MAX_FUNCTIONAL_PROTEINS, FIXED_EMBEDDING_DIM)))
        final_classification_stack.append(np.zeros((MAX_CLASSIFICATION_PROTEINS, FIXED_EMBEDDING_DIM)))

# Convert lists to unified numpy arrays
final_virus_names = np.array(final_virus_name_list)
final_membrane_embeds = np.stack(final_membrane_stack)       # Shape: (Total_N, 4, 1152)
final_nuclear_embeds = np.stack(final_nuclear_stack)         # Shape: (Total_N, 5, 1152)
final_functional_embeds = np.stack(final_functional_stack)    # Shape: (Total_N, 6, 1152)
final_classification_embeds = np.stack(final_classification_stack)  # Shape: (Total_N, 1, 1152)

# ---------------------------- Generate Aggregated Global Embedding Vectors ----------------------------
# Step 1: Average embeddings across all proteins within each category
avg_membrane_per_virus = final_membrane_embeds.mean(axis=1)    # (N, 1152)
avg_nuclear_per_virus = final_nuclear_embeds.mean(axis=1)      # (N, 1152)
avg_functional_per_virus = final_functional_embeds.mean(axis=1)# (N, 1152)

# Step 2: Fuse three functional categories into a single composite embedding
composite_category_embedding = (avg_membrane_per_virus + avg_nuclear_per_virus + avg_functional_per_virus) / 3.0

# Step 3: Remove redundant single-protein dimension for classification embeddings
single_class_embedding = final_classification_embeds.reshape(len(final_virus_names), -1)  # (N, 1152)

# Step 4: Concatenate composite category embedding + classification embedding, perform standardization
raw_concat_embedding = np.concatenate([composite_category_embedding, single_class_embedding], axis=1)  # (N, 2304)
feature_scaler = StandardScaler()
normalized_global_embedding = feature_scaler.fit_transform(raw_concat_embedding)

# Step 5: Save all embedding outputs to compressed NPZ file
np.savez(
    "virus_three_embeddings_esmc600m.npz",
    virus_names=final_virus_names,
    composite_category_embedding=composite_category_embedding,
    classification_embedding=single_class_embedding,
    normalized_global_embedding=normalized_global_embedding
)

# Print output summary information
print("=" * 50)
print("Output Tensor Shapes Summary:")
print(f"  virus_names                : {final_virus_names.shape}")
print(f"  composite_category_embedding: {composite_category_embedding.shape}")
print(f"  classification_embedding   : {single_class_embedding.shape}")
print(f"  normalized_global_embedding: {normalized_global_embedding.shape}")
print("Embedding file saved as: virus_three_embeddings_esmc600m.npz")