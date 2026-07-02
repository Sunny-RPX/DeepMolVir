# src/train_regressor.py
# Regression training (pIC50 prediction) with 5-fold cross-validation

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import CombinedModelGRU
from data_utils import MultiModalDataset, collate_fn, load_and_prepare_data

# ------------------------ Default hyperparameters ------------------------
DEFAULTS = {
    'virus_dim': 2304,
    'ecfp_dim': 2048,
    'unimol_dim': 512,
    'kagat_out_dim': 256,
    'node_in_dim': 100,
    'edge_in_dim': 5,
    'hidden_dim': 256,
    'grid_size': 8,
    'heads': 4,
    'layer_num': 2,
    'pooling': 'avg',
    'gru_hidden_dim': 256,
    'gru_num_layers': 2,
    'dropout': 0.2,
    'batch_size': 64,
    'epochs': 50,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'grad_clip': 0.5,
    'patience': 10,
    'n_folds': 5,
    'seed': 42,
}

def train_one_fold(train_dataset, val_dataset, fold_idx, args, device):
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)

    model = CombinedModelGRU(
        virus_dim=DEFAULTS['virus_dim'],
        ecfp_dim=DEFAULTS['ecfp_dim'],
        unimol_dim=DEFAULTS['unimol_dim'],
        kagat_dim=DEFAULTS['kagat_out_dim'],
        gru_hidden_dim=DEFAULTS['gru_hidden_dim'],
        gru_num_layers=DEFAULTS['gru_num_layers'],
        dropout=DEFAULTS['dropout'],
        node_in_dim=DEFAULTS['node_in_dim'],
        edge_in_dim=DEFAULTS['edge_in_dim'],
        hidden_dim=DEFAULTS['hidden_dim'],
        grid_size=DEFAULTS['grid_size'],
        heads=DEFAULTS['heads'],
        layer_num=DEFAULTS['layer_num'],
        pooling=DEFAULTS['pooling']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for g, virus, ecfp, unimol, y in train_loader:
            g = g.to(device)
            virus = virus.to(device)
            ecfp = ecfp.to(device)
            unimol = unimol.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(g, virus, ecfp, unimol)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULTS['grad_clip'])
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for g, virus, ecfp, unimol, y in val_loader:
                g = g.to(device)
                virus = virus.to(device)
                ecfp = ecfp.to(device)
                unimol = unimol.to(device)
                pred = model(g, virus, ecfp, unimol)
                val_preds.extend(pred.cpu().numpy())
                val_true.extend(y.numpy())
        val_rmse = np.sqrt(mean_squared_error(val_true, val_preds))
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
        if epoch % 10 == 0:
            print(f"  Fold {fold_idx+1}, Epoch {epoch+1}/{args.epochs}, "
                  f"Loss: {total_loss/len(train_loader):.4f}, Val RMSE: {val_rmse:.4f}")
        if no_improve >= args.patience:
            break

    model.load_state_dict(best_state)
    model_dir = os.path.join(args.output_dir, 'regressor')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(best_state, os.path.join(model_dir, f'fold_{fold_idx}.pth'))
    return model, best_val_rmse

def evaluate_regressor(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=collate_fn)
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for g, virus, ecfp, unimol, y in loader:
            g = g.to(device)
            virus = virus.to(device)
            ecfp = ecfp.to(device)
            unimol = unimol.to(device)
            pred = model(g, virus, ecfp, unimol)
            all_preds.extend(pred.cpu().numpy())
            all_true.extend(y.numpy())
    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    mse = mean_squared_error(all_true, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_true, all_preds)
    r2 = r2_score(all_true, all_preds)
    pearson_corr, _ = pearsonr(all_true, all_preds)
    spearman_corr, _ = spearmanr(all_true, all_preds)
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Pearson': pearson_corr,
        'Spearman': spearman_corr
    }
    return metrics, all_true, all_preds

def main():
    parser = argparse.ArgumentParser(description='Train DeepMolVir Regressor (pIC50)')
    parser.add_argument('--smiles_file', type=str, required=True,
                        help='Path to Database_smiles.xlsx')
    parser.add_argument('--virus_npz', type=str, required=True,
                        help='Path to virus embedding .npz')
    parser.add_argument('--unimol_npy', type=str, required=True,
                        help='Path to UniMol CLS .npy')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='Directory containing train/val/test index .npy files')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory to save models and results')
    parser.add_argument('--batch_size', type=int, default=DEFAULTS['batch_size'])
    parser.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    parser.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    parser.add_argument('--weight_decay', type=float, default=DEFAULTS['weight_decay'])
    parser.add_argument('--patience', type=int, default=DEFAULTS['patience'])
    parser.add_argument('--n_folds', type=int, default=DEFAULTS['n_folds'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    torch.manual_seed(DEFAULTS['seed'])
    np.random.seed(DEFAULTS['seed'])
    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("Loading and preparing data...")
    virus_std, ecfp_std, unimol_std, graphs, y, valid_idx, scalers = load_and_prepare_data(
        args.smiles_file, args.virus_npz, args.unimol_npy, task='regression'
    )

    test_idx = np.load(os.path.join(args.splits_dir, 'test_indices.npy')).tolist()
    cv_splits = []
    for fold in range(args.n_folds):
        train_idx = np.load(os.path.join(args.splits_dir, f'train_indices_fold_{fold}.npy')).tolist()
        val_idx = np.load(os.path.join(args.splits_dir, f'val_indices_fold_{fold}.npy')).tolist()
        cv_splits.append((train_idx, val_idx))

    print(f"Test set size: {len(test_idx)}")
    virus_test = virus_std[test_idx]
    ecfp_test = ecfp_std[test_idx]
    unimol_test = unimol_std[test_idx]
    graphs_test = [graphs[i] for i in test_idx]
    y_test = y[test_idx]
    test_dataset = MultiModalDataset(virus_test, ecfp_test, unimol_test, graphs_test, y_test)

    all_virus = virus_std
    all_ecfp = ecfp_std
    all_unimol = unimol_std
    all_graphs = graphs
    all_targets = y

    print("\n========== Starting 5-Fold Cross-Validation (Regression) ==========")
    fold_val_rmses = []
    fold_test_metrics = []

    for fold_idx, (train_idx_fold, val_idx_fold) in enumerate(cv_splits):
        print(f"  Fold {fold_idx+1}/{args.n_folds}")
        train_ds = MultiModalDataset(
            all_virus[train_idx_fold], all_ecfp[train_idx_fold],
            all_unimol[train_idx_fold],
            [all_graphs[i] for i in train_idx_fold],
            all_targets[train_idx_fold]
        )
        val_ds = MultiModalDataset(
            all_virus[val_idx_fold], all_ecfp[val_idx_fold],
            all_unimol[val_idx_fold],
            [all_graphs[i] for i in val_idx_fold],
            all_targets[val_idx_fold]
        )
        model, best_val_rmse = train_one_fold(train_ds, val_ds, fold_idx, args, device)
        fold_val_rmses.append(best_val_rmse)

        test_metrics, _, _ = evaluate_regressor(model, test_dataset, device, args.batch_size)
        fold_test_metrics.append(test_metrics)
        print(f"    Val RMSE: {best_val_rmse:.3f} | Test RMSE: {test_metrics['RMSE']:.3f}, R2: {test_metrics['R2']:.4f}")

    mean_val_rmse = np.mean(fold_val_rmses)
    std_val_rmse = np.std(fold_val_rmses)
    mean_test = {}
    std_test = {}
    for metric in fold_test_metrics[0].keys():
        values = [m[metric] for m in fold_test_metrics]
        mean_test[metric] = np.mean(values)
        std_test[metric] = np.std(values)

    print("\n========== Final Results (Regression) ==========")
    print(f"Validation RMSE (mean +/- std): {mean_val_rmse:.4f} +/- {std_val_rmse:.4f}")
    print("Test set metrics (mean +/- std):")
    for k in mean_test:
        print(f"  {k}: {mean_test[k]:.4f} +/- {std_test[k]:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'regressor_results.txt'), 'w') as f:
        f.write("DeepMolVir Regressor 5-Fold CV Results\n")
        f.write(f"Validation RMSE (mean +/- std): {mean_val_rmse:.4f} +/- {std_val_rmse:.4f}\n")
        for k in mean_test:
            f.write(f"Test {k}: {mean_test[k]:.4f} +/- {std_test[k]:.4f}\n")

    # Save scalers
    scaler_dir = os.path.join(args.output_dir, 'scalers')
    os.makedirs(scaler_dir, exist_ok=True)
    with open(os.path.join(scaler_dir, 'virus_scaler.pkl'), 'wb') as f:
        pickle.dump(scalers[0], f)
    with open(os.path.join(scaler_dir, 'ecfp_scaler.pkl'), 'wb') as f:
        pickle.dump(scalers[1], f)
    with open(os.path.join(scaler_dir, 'unimol_scaler.pkl'), 'wb') as f:
        pickle.dump(scalers[2], f)
    print("Scalers saved.")

if __name__ == '__main__':
    main()