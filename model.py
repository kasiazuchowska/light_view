import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = str(__import__("pathlib").Path(__file__).parent / "model.pt")

SPD_DIM = 401
LATENT_DIM = 16

# CCT is conditioned in log-space, normalized to [0, 1] over the dataset range
_CCT_LOG_MIN = np.log(1630.0)
_CCT_LOG_MAX = np.log(18273.0)


def normalize_cct(cct) -> np.ndarray:
    return (np.log(np.asarray(cct, dtype=float)) - _CCT_LOG_MIN) / (_CCT_LOG_MAX - _CCT_LOG_MIN)


class ConditionalVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(SPD_DIM + 1, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, LATENT_DIM)
        self.fc_logvar = nn.Linear(128, LATENT_DIM)

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, SPD_DIM),
            nn.Sigmoid(),
        )

    def encode(self, spd: torch.Tensor, cct_norm: torch.Tensor):
        x = torch.cat([spd, cct_norm.unsqueeze(1)], dim=1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, cct_norm: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, cct_norm.unsqueeze(1)], dim=1)
        return self.decoder(x)

    def forward(self, spd: torch.Tensor, cct_norm: torch.Tensor):
        mu, logvar = self.encode(spd, cct_norm)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, cct_norm), mu, logvar

    @staticmethod
    def sample_z() -> np.ndarray:
        """Sample a random latent direction (unit-scale)."""
        return torch.randn(LATENT_DIM).numpy()

    def decode_z(self, z: np.ndarray, cct: float) -> np.ndarray:
        """Decode an explicit latent vector z (shape: LATENT_DIM,) for the given CCT."""
        self.eval()
        with torch.no_grad():
            cct_norm = torch.tensor([normalize_cct(cct)], dtype=torch.float32)
            z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
            spd = self.decode(z_t, cct_norm)
        return spd[0].numpy()
