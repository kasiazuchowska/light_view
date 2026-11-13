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


def _vae_loss(recon, target, mu, logvar, beta):
    recon_loss = F.mse_loss(recon, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss.item(), kl.item()


def train():
    print("Loading dataset...")
    ccts, spds = load_dataset()
    print(f"  {len(ccts)} SPDs loaded, CCT range: {ccts.min():.0f}–{ccts.max():.0f} K")

    ccts_norm = normalize_cct(ccts).astype(np.float32)
    spds = spds.astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(spds), torch.from_numpy(ccts_norm))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ConditionalVAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        beta = BETA_MAX * min(1.0, epoch / ANNEAL_EPOCHS)
        model.train()
        total_loss = 0.0
        for spd_batch, cct_batch in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(spd_batch, cct_batch)
            loss, _, _ = _vae_loss(recon, spd_batch, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(spd_batch)

        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}/{EPOCHS}  loss={total_loss / len(dataset):.5f}  beta={beta:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
