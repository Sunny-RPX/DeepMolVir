#!/usr/bin/env python
# predict_regressor.py
# Predict pIC50 values using the 5-fold ensemble.
# This script is located in the src/ directory.
# Input files are expected in ../data/case1/ by default.

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import dgl
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to sys.path to import src modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.models import CombinedModelGRU
from src.data_utils import clip_percentile, smiles_to_graph

# --------------------------- Default Paths ---------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'case1')
DEFAULT_CSV = os.path.join(DATA_DIR, "casestudy1.csv")
DEFAULT_ECFP = os.path.join(DATA_DIR, "casestudy1_ecfp4_2048.npy")
DEFAULT_UNIMOL = os.path.join(DATA_DIR, "casestudy1_unimol_embeddings.npy")
DEFAULT_VIRUS = os.path.join(DATA_DIR, "virus_three_embeddings_esmc600m.npz")
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "regressor")
DEFAULT_SCALER_DIR = os.path.join(PROJECT_ROOT, "models", "scalers")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "casestudy1_regressor_predictions.csv")

# Model hyperparameters (must match training)
VIRUS_DIM = 2304
ECFP_DIM = 2048
UNIMOL_DIM = 512
KAGAT_OUT_DIM = 256
NODE_IN_DIM = 100
EDGE_IN_DIM = 5
HIDDEN_DIM = 256
GRID_SIZE = 8
HEADS = 4
LAYER_NUM = 2
POOLING = 'avg'
GRU_HIDDEN_DIM = 256
GRU_NUM_LAYERS = 2
GRU_DROPOUT = 0.2
N_FOLDS = 5
BATCH_SIZE = 64

# --------------------------- Helper ---------------------------
def pIC50_to_nM(pic50):
    return 10.0 ** (9.0 - pic50)

# --------------------------- Prediction Dataset & Collate ---------------------------
class PredictionDataset(Dataset):
    def __init__(self, graphs, virus_feat, ecfp_feat, unimol_feat):
        self.graphs = graphs
        self.virus = torch.FloatTensor(virus_feat)
        self.ecfp = torch.FloatTensor(ecfp_feat)
        self.unimol = torch.FloatTensor(unimol_feat)
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return (self.graphs[idx], self.virus[idx], self.ecfp[idx], self.unimol[idx])

def collate_fn_pred(batch):
    graphs, virus_list, ecfp_list, unimol_list = zip(*batch)
    batched_graph = dgl.batch(graphs)
    virus = torch.stack(virus_list)
    ecfp = torch.stack(ecfp_list)
    unimol = torch.stack(unimol_list)
    return batched_graph, virus, ecfp, unimol

# --------------------------- Main ---------------------------
def main():
    parser = argparse.ArgumentParser(description='DeepMolVir Regressor Prediction (pIC50)')
    parser.add_argument('--csv', type=str, default=DEFAULT_CSV,
                        help='Input CSV with SMILES and vVirus name columns')
    parser.add_argument('--ecfp', type=str, default=DEFAULT_ECFP,
                        help='Precomputed ECFP4 features (.npy)')
    parser.add_argument('--unimol', type=str, default=DEFAULT_UNIMOL,
                        help='Precomputed UniMol CLS features (.npy)')
    parser.add_argument('--virus_npz', type=str, default=DEFAULT_VIRUS,
                        help='Virus embedding file (.npz)')
    parser.add_argument('--model_dir', type=str, default=DEFAULT_MODEL_DIR,
                        help='Directory containing fold_0.pth ... fold_4.pth')
    parser.add_argument('--scaler_dir', type=str, default=DEFAULT_SCALER_DIR,
                        help='Directory with virus_scaler.pkl, ecfp_scaler.pkl, unimol_scaler.pkl')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help='Output CSV file')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load input data
    print("Loading CSV...")
    df = pd.read_csv(args.csv)
    if 'Smiles' not in df.columns:
        raise KeyError("CSV must contain a 'Smiles' column")
    if 'vVirus name' not in df.columns:
        raise KeyError("CSV must contain a 'vVirus name' column")
    smiles_list = df['Smiles'].tolist()
    virus_names = df['vVirus name'].str.strip().tolist()
    n_samples = len(smiles_list)
    print(f"Loaded {n_samples} molecules.")

    # 2. Load precomputed features
    print("Loading ECFP and UniMol features...")
    ecfp_data = np.load(args.ecfp)
    unimol_data = np.load(args.unimol)
    assert ecfp_data.shape[0] == n_samples and unimol_data.shape[0] == n_samples, \
        "Feature shape mismatch with number of SMILES."

    # 3. Load virus embeddings
    print("Loading virus embeddings...")
    virus_npz = np.load(args.virus_npz, allow_pickle=True)
    virus_names_db = virus_npz['virus_names']
    virus_embeddings_db = virus_npz['virus_embedding']
    virus_emb_dict = {name: virus_embeddings_db[i] for i, name in enumerate(virus_names_db)}

    virus_arr = []
    for vname in virus_names:
        if vname in virus_emb_dict:
            virus_arr.append(virus_emb_dict[vname])
        else:
            print(f"Warning: Virus '{vname}' not found, using zero vector")
            virus_arr.append(np.zeros(VIRUS_DIM))
    virus_arr = np.stack(virus_arr)

    # 4. Load scalers and preprocess
    print("Loading scalers...")
    with open(os.path.join(args.scaler_dir, 'virus_scaler.pkl'), 'rb') as f:
        scaler_virus = pickle.load(f)
    with open(os.path.join(args.scaler_dir, 'ecfp_scaler.pkl'), 'rb') as f:
        scaler_ecfp = pickle.load(f)
    with open(os.path.join(args.scaler_dir, 'unimol_scaler.pkl'), 'rb') as f:
        scaler_unimol = pickle.load(f)

    virus_arr = clip_percentile(virus_arr, 1, 99)
    ecfp_data = clip_percentile(ecfp_data, 1, 99)
    unimol_data = clip_percentile(unimol_data, 1, 99)

    virus_std = scaler_virus.transform(virus_arr)
    virus_std = np.clip(virus_std, -5, 5)
    ecfp_std = scaler_ecfp.transform(ecfp_data)
    ecfp_std = np.clip(ecfp_std, -5, 5)
    unimol_std = scaler_unimol.transform(unimol_data)
    unimol_std = np.clip(unimol_std, -5, 5)

    # 5. Build molecular graphs
    print("Building molecular graphs (this may take a while)...")
    graphs = [smiles_to_graph(smi) for smi in smiles_list]

    # 6. Load ensemble models
    print("Loading models...")
    models = []
    for fold in range(N_FOLDS):
        weight_file = os.path.join(args.model_dir, f'fold_{fold}.pth')
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"Model file {weight_file} not found.")
        model = CombinedModelGRU(
            virus_dim=VIRUS_DIM,
            ecfp_dim=ECFP_DIM,
            unimol_dim=UNIMOL_DIM,
            kagat_dim=KAGAT_OUT_DIM,
            gru_hidden_dim=GRU_HIDDEN_DIM,
            gru_num_layers=GRU_NUM_LAYERS,
            dropout=GRU_DROPOUT,
            node_in_dim=NODE_IN_DIM,
            edge_in_dim=EDGE_IN_DIM,
            hidden_dim=HIDDEN_DIM,
            grid_size=GRID_SIZE,
            heads=HEADS,
            layer_num=LAYER_NUM,
            pooling=POOLING
        ).to(device)
        model.load_state_dict(torch.load(weight_file, map_location=device))
        model.eval()
        models.append(model)

    # 7. Run inference
    pred_dataset = PredictionDataset(graphs, virus_std, ecfp_std, unimol_std)
    pred_loader = DataLoader(pred_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn_pred)

    all_preds = []
    for model in models:
        fold_preds = []
        with torch.no_grad():
            for g, virus, ecfp, unimol in pred_loader:
                g = g.to(device)
                virus = virus.to(device)
                ecfp = ecfp.to(device)
                unimol = unimol.to(device)
                pred = model(g, virus, ecfp, unimol)
                fold_preds.append(pred.cpu().numpy())
        fold_preds = np.concatenate(fold_preds)
        all_preds.append(fold_preds)

    all_preds = np.array(all_preds)  # (5, n_samples)
    mean_pIC50 = np.mean(all_preds, axis=0)
    std_pIC50 = np.std(all_preds, axis=0)
    mean_nM = pIC50_to_nM(mean_pIC50)

    # 8. Save results
    df['predicted_pIC50'] = mean_pIC50
    df['pIC50_std'] = std_pIC50
    df['predicted_nM'] = mean_nM

    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

    print("\nFirst 5 predictions:")
    print(df[['Smiles', 'vVirus name', 'predicted_pIC50', 'predicted_nM']].head())

if __name__ == '__main__':
    main()