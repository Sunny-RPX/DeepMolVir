# src/data_utils.py
# Data loading, preprocessing, graph construction, fingerprint generation, dataset

import os
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import dgl
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# ------------------------ Constants ------------------------
NODE_IN_DIM = 100
EDGE_IN_DIM = 5

# ------------------------ Utility Functions ------------------------
def clip_percentile(x, low=1, high=99):
    p_low = np.percentile(x, low, axis=0, keepdims=True)
    p_high = np.percentile(x, high, axis=0, keepdims=True)
    return np.clip(x, p_low, p_high)

def ecfp4_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return np.array(fp)

def bond_type_to_onehot(bond):
    btype = bond.GetBondType()
    if btype == Chem.rdchem.BondType.SINGLE:
        idx = 0
    elif btype == Chem.rdchem.BondType.DOUBLE:
        idx = 1
    elif btype == Chem.rdchem.BondType.TRIPLE:
        idx = 2
    elif btype == Chem.rdchem.BondType.AROMATIC:
        idx = 3
    else:
        idx = 4
    onehot = np.zeros(EDGE_IN_DIM)
    onehot[idx] = 1.0
    return onehot

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        g = dgl.graph([(0, 0)], num_nodes=1)
        g.ndata['feat'] = torch.zeros(1, NODE_IN_DIM, dtype=torch.float)
        g.edata['feat'] = torch.zeros(1, EDGE_IN_DIM, dtype=torch.float)
        return g

    num_atoms = mol.GetNumAtoms()
    atom_feats = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        feat = np.zeros(NODE_IN_DIM)
        if 1 <= atomic_num <= NODE_IN_DIM:
            feat[atomic_num - 1] = 1
        atom_feats.append(feat)

    # Convert list to numpy array before tensor creation to avoid warning
    node_feats = torch.tensor(np.array(atom_feats), dtype=torch.float)

    edges = []
    edge_feats = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        edges.append((u, v))
        edges.append((v, u))
        ft = bond_type_to_onehot(bond)
        edge_feats.append(ft)
        edge_feats.append(ft.copy())
    if len(edges) == 0:
        edges = [(0, 0)]
        edge_feats = [np.zeros(EDGE_IN_DIM)]
    g = dgl.graph(edges, num_nodes=num_atoms)
    g.ndata['feat'] = node_feats
    g.edata['feat'] = torch.tensor(np.array(edge_feats), dtype=torch.float)
    return g

def nm_to_pIC50(nm_values):
    nm_values = np.asarray(nm_values, dtype=float)
    nm_values = np.clip(nm_values, 1e-6, None)
    return 9.0 - np.log10(nm_values)

def load_and_prepare_data(smiles_file, virus_npz, unimol_npy, task='classification'):
    """
    Load SMILES, virus embeddings, UniMol features, compute ECFP4,
    build graphs, and return standardized features and labels.
    Returns:
        virus_std, ecfp_std, unimol_std, graphs, targets, valid_indices, scalers
    """
    # Read SMILES
    df = pd.read_excel(smiles_file)
    # Virus embeddings
    virus_data = np.load(virus_npz, allow_pickle=True)
    virus_names = virus_data['virus_names']
    virus_embeddings = virus_data['virus_embedding']
    virus_dict = {name: virus_embeddings[i] for i, name in enumerate(virus_names)}
    df['vVirus name'] = df['vVirus name'].str.strip()
    df['virus_emb'] = df['vVirus name'].map(virus_dict)
    df = df.dropna(subset=['virus_emb']).reset_index(drop=True)

    # Compute ECFP4 and filter valid molecules
    ecfp_list = []
    valid_idx = []
    for i, row in df.iterrows():
        smi = row['Smiles']
        fp = ecfp4_fingerprint(smi)
        if np.sum(fp) == 0:
            continue
        ecfp_list.append(fp)
        valid_idx.append(i)
    df = df.iloc[valid_idx].reset_index(drop=True)
    ecfp_arr = np.stack(ecfp_list)

    # Virus features
    virus_arr = np.stack(df['virus_emb'].values)
    virus_arr = np.nan_to_num(virus_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # UniMol features
    unimol_full = np.load(unimol_npy)
    unimol_arr = unimol_full[valid_idx]

    # Clip percentiles
    virus_arr = clip_percentile(virus_arr, 1, 99)
    ecfp_arr = clip_percentile(ecfp_arr, 1, 99)
    unimol_arr = clip_percentile(unimol_arr, 1, 99)

    # Standardize
    scaler_virus = StandardScaler()
    virus_std = scaler_virus.fit_transform(virus_arr)
    virus_std = np.clip(virus_std, -5, 5)

    scaler_ecfp = StandardScaler()
    ecfp_std = scaler_ecfp.fit_transform(ecfp_arr)
    ecfp_std = np.clip(ecfp_std, -5, 5)

    scaler_unimol = StandardScaler()
    unimol_std = scaler_unimol.fit_transform(unimol_arr)
    unimol_std = np.clip(unimol_std, -5, 5)

    # Build graphs
    graphs = [smiles_to_graph(smi) for smi in df['Smiles']]

    # Targets
    if task == 'classification':
        y = (df['nM-value'] <= 5000).astype(int).values
    else:  # regression
        y = nm_to_pIC50(df['nM-value'].values)
    y = y[valid_idx] if task == 'classification' else y

    scalers = (scaler_virus, scaler_ecfp, scaler_unimol)

    return virus_std, ecfp_std, unimol_std, graphs, y, valid_idx, scalers

# ------------------------ Dataset and Collate ------------------------
class MultiModalDataset(Dataset):
    def __init__(self, virus_feat, ecfp_feat, unimol_feat, graphs, targets):
        self.virus = torch.FloatTensor(virus_feat)
        self.ecfp = torch.FloatTensor(ecfp_feat)
        self.unimol = torch.FloatTensor(unimol_feat)
        self.graphs = graphs
        self.targets = torch.FloatTensor(targets)
        self.n_samples = len(targets)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (self.graphs[idx], self.virus[idx], self.ecfp[idx],
                self.unimol[idx], self.targets[idx])

def collate_fn(batch):
    graphs, virus_list, ecfp_list, unimol_list, y_list = zip(*batch)
    batched_graph = dgl.batch(graphs)
    virus = torch.stack(virus_list)
    ecfp = torch.stack(ecfp_list)
    unimol = torch.stack(unimol_list)
    y = torch.stack(y_list)
    return batched_graph, virus, ecfp, unimol, y