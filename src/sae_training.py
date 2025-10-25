"""Sparse Autoencoder (SAE) training for interpretable feature discovery."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformer_lens import HookedTransformer
from typing import Optional, Tuple, List, Dict, Callable
import logging
from pathlib import Path
from tqdm.auto import tqdm
import einops

from utils import get_mlp_activations

logger = logging.getLogger(__name__)


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for learning interpretable features from activations.

    Architecture:
        activation (d_model) -> encoder -> latent (d_hidden) -> decoder -> reconstruction (d_model)

    Training objective:
        L = ||x - x_hat||^2 + lambda * ||z||_1
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        sparsity_coef: float = 1e-3,
        tied_weights: bool = False,
    ):
        """
        Initialize SAE.

        Args:
            d_model: Input/output dimension
            d_hidden: Hidden dimension (typically 4-8x d_model)
            sparsity_coef: L1 penalty coefficient
            tied_weights: If True, decoder = encoder.T
        """
        super().__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.sparsity_coef = sparsity_coef
        self.tied_weights = tied_weights

        # Encoder
        self.encoder = nn.Linear(d_model, d_hidden, bias=True)
        nn.init.xavier_uniform_(self.encoder.weight)

        # Decoder
        if tied_weights:
            self.decoder = None  # Use encoder.weight.T
        else:
            self.decoder = nn.Linear(d_hidden, d_model, bias=True)
            nn.init.xavier_uniform_(self.decoder.weight)

        # Output bias (for centering)
        self.bias = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return F.relu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        if self.tied_weights:
            return F.linear(z, self.encoder.weight.T, self.bias)
        else:
            return self.decoder(z) + self.bias

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input activations [batch, ..., d_model]

        Returns:
            Tuple of (reconstruction, latent_activations)
        """
        # Flatten batch dimensions
        original_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)

        # Encode
        z = self.encode(x_flat)

        # Decode
        x_hat = self.decode(z)

        # Reshape back
        x_hat = x_hat.reshape(original_shape)
        z = z.reshape(*original_shape[:-1], self.d_hidden)

        return x_hat, z

    def loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute SAE loss.

        Args:
            x: Original activations
            x_hat: Reconstructed activations
            z: Latent activations

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_hat, x)

        # Sparsity loss (L1)
        sparsity_loss = z.abs().mean()

        # Total loss
        total_loss = recon_loss + self.sparsity_coef * sparsity_loss

        metrics = {
            "loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "l0": (z > 0).float().mean().item(),  # Average number of active features
        }

        return total_loss, metrics


class ActivationDataset(Dataset):
    """Dataset of cached activations for SAE training."""

    def __init__(self, activations: torch.Tensor):
        """
        Initialize dataset.

        Args:
            activations: Tensor of shape [n_samples, seq_len, d_model]
        """
        self.activations = activations

    def __len__(self) -> int:
        return len(self.activations)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.activations[idx]


def collect_activations(
    model: HookedTransformer,
    prompts: List[str],
    layer: int,
    component: str = "mlp_out",
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Collect activations from a model for SAE training.

    Args:
        model: HookedTransformer model
        prompts: List of prompts to run
        layer: Layer to extract from
        component: Component to extract
        batch_size: Batch size for processing

    Returns:
        Tensor of activations [n_prompts, seq_len, d_model]
    """
    all_activations = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Collecting activations"):
        batch_prompts = prompts[i:i + batch_size]

        # Run batch
        _, cache = model.run_with_cache(batch_prompts)
        acts = cache[component, layer].cpu()

        all_activations.append(acts)

    return torch.cat(all_activations, dim=0)


def train_sae(
    sae: SparseAutoencoder,
    train_loader: DataLoader,
    n_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda",
    log_interval: int = 100,
) -> List[Dict[str, float]]:
    """
    Train a sparse autoencoder.

    Args:
        sae: SparseAutoencoder model
        train_loader: DataLoader of activations
        n_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        log_interval: Steps between logging

    Returns:
        List of training metrics per step
    """
    sae = sae.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    training_log = []

    for epoch in range(n_epochs):
        epoch_metrics = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")

        for step, batch in enumerate(progress):
            batch = batch.to(device)

            # Forward pass
            x_hat, z = sae(batch)
            loss, metrics = sae.loss(batch, x_hat, z)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log
            epoch_metrics.append(metrics)
            training_log.append(metrics)

            if step % log_interval == 0:
                progress.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "recon": f"{metrics['recon_loss']:.4f}",
                    "l0": f"{metrics['l0']:.1f}",
                })

        # Epoch summary
        avg_loss = sum(m["loss"] for m in epoch_metrics) / len(epoch_metrics)
        avg_l0 = sum(m["l0"] for m in epoch_metrics) / len(epoch_metrics)

        logger.info(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, L0: {avg_l0:.1f}")

    return training_log


def save_sae(sae: SparseAutoencoder, path: str, metadata: Optional[Dict] = None) -> None:
    """
    Save SAE model and metadata.

    Args:
        sae: SparseAutoencoder model
        path: Save path
        metadata: Optional metadata dict
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": sae.state_dict(),
        "config": {
            "d_model": sae.d_model,
            "d_hidden": sae.d_hidden,
            "sparsity_coef": sae.sparsity_coef,
            "tied_weights": sae.tied_weights,
        },
        "metadata": metadata or {},
    }

    torch.save(checkpoint, save_path)
    logger.info(f"SAE saved to {save_path}")


def load_sae(path: str, device: str = "cpu") -> SparseAutoencoder:
    """
    Load SAE model from checkpoint.

    Args:
        path: Path to checkpoint
        device: Device to load on

    Returns:
        Loaded SparseAutoencoder
    """
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]

    sae = SparseAutoencoder(**config)
    sae.load_state_dict(checkpoint["state_dict"])
    sae.to(device)

    logger.info(f"SAE loaded from {path}")

    return sae


def get_top_activating_examples(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    prompts: List[str],
    feature_idx: int,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Find examples that most activate a specific SAE feature.

    Args:
        sae: Trained SparseAutoencoder
        activations: Input activations [n_examples, seq_len, d_model]
        prompts: Corresponding prompts
        feature_idx: Feature index to analyze
        top_k: Number of top examples to return

    Returns:
        List of (prompt, activation_value) tuples
    """
    sae.eval()

    all_activations = []

    with torch.no_grad():
        for act in activations:
            _, z = sae(act.to(next(sae.parameters()).device))
            # Max activation for this feature across sequence
            max_act = z[:, feature_idx].max().item()
            all_activations.append(max_act)

    # Get top-k
    top_indices = torch.tensor(all_activations).topk(top_k).indices

    results = [(prompts[idx], all_activations[idx]) for idx in top_indices]

    return results
