import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from data_loader import load_dataset
from model import ConditionalVAE, normalize_cct, MODEL_PATH

EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
BETA_MAX = 0.001   # final KL weight — low enough to keep z informative
ANNEAL_EPOCHS = 100  # linearly warm up KL from 0 to BETA_MAX over this many epochs

SEED = 42
TEST_FRACTION = 0.15
PROJECT_DIR = Path(__file__).parent
TEST_INDICES_PATH = PROJECT_DIR / "data" / "test_indices.json"
LOG_PATH = PROJECT_DIR / "training_log.csv"


def _vae_loss(recon, target, mu, logvar, beta):
    recon_loss = F.mse_loss(recon, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss.item(), kl.item()


def _evaluate(model, loader, beta):
    """Return per-sample mean (loss, recon, kl) over the loader, without updating the model."""
    model.eval()
    totals = np.zeros(3)
    n = 0
    with torch.no_grad():
        for spd_batch, cct_batch in loader:
            recon, mu, logvar = model(spd_batch, cct_batch)
            loss, recon_loss, kl = _vae_loss(recon, spd_batch, mu, logvar, beta)
            totals += np.array([loss.item(), recon_loss, kl]) * len(spd_batch)
            n += len(spd_batch)
    return totals / n


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading dataset...")
    ccts, spds = load_dataset()
    print(f"  {len(ccts)} SPDs loaded, CCT range: {ccts.min():.0f}–{ccts.max():.0f} K")

    ccts_norm = normalize_cct(ccts).astype(np.float32)
    spds = spds.astype(np.float32)

    # Random train/test split; indices refer to the order returned by load_dataset()
    perm = np.random.permutation(len(ccts))
    n_test = int(round(TEST_FRACTION * len(ccts)))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    TEST_INDICES_PATH.write_text(json.dumps(test_idx.tolist()))
    print(f"  train: {len(train_idx)} SPDs, test: {len(test_idx)} SPDs (indices saved to {TEST_INDICES_PATH})")

    train_set = TensorDataset(torch.from_numpy(spds[train_idx]), torch.from_numpy(ccts_norm[train_idx]))
    test_set = TensorDataset(torch.from_numpy(spds[test_idx]), torch.from_numpy(ccts_norm[test_idx]))
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = ConditionalVAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    log_file = open(LOG_PATH, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["epoch", "beta", "train_loss", "train_recon", "train_kl", "test_loss", "test_recon", "test_kl"])

    print(f"Training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        beta = BETA_MAX * min(1.0, epoch / ANNEAL_EPOCHS)
        model.train()
        train_totals = np.zeros(3)
        for spd_batch, cct_batch in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(spd_batch, cct_batch)
            loss, recon_loss, kl = _vae_loss(recon, spd_batch, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            train_totals += np.array([loss.item(), recon_loss, kl]) * len(spd_batch)

        train_loss, train_recon, train_kl = train_totals / len(train_set)
        test_loss, test_recon, test_kl = _evaluate(model, test_loader, beta)
        log.writerow([epoch, beta, train_loss, train_recon, train_kl, test_loss, test_recon, test_kl])

        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}/{EPOCHS}  train_loss={train_loss:.5f}  test_loss={test_loss:.5f}  beta={beta:.4f}")

    log_file.close()
    print(f"Training log saved to {LOG_PATH}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
