"""Utility functions for model loading, activation extraction, and common operations."""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import Optional, Tuple, List, Dict, Callable
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the project."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def load_model(
    model_name: str,
    device: Optional[str] = None,
    center_unembed: bool = True,
    center_writing_weights: bool = True,
    fold_ln: bool = True,
    refactor_factored_attn_matrices: bool = True,
) -> HookedTransformer:
    """
    Load a model with TransformerLens.

    Args:
        model_name: Model identifier (e.g., 'gpt2-medium', 'meta-llama/Llama-3.2-1B')
        device: Device to load model on. Auto-detected if None.
        center_unembed: Center the unembedding weights
        center_writing_weights: Center weights that write to residual stream
        fold_ln: Fold LayerNorm into weights
        refactor_factored_attn_matrices: Refactor attention matrices for efficiency

    Returns:
        Loaded HookedTransformer model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading {model_name} on {device}...")

    model = HookedTransformer.from_pretrained(
        model_name,
        center_unembed=center_unembed,
        center_writing_weights=center_writing_weights,
        fold_ln=fold_ln,
        refactor_factored_attn_matrices=refactor_factored_attn_matrices,
        device=device,
    )

    logger.info(f"Model loaded: {model.cfg.n_layers}L, {model.cfg.n_heads}H, d_model={model.cfg.d_model}")

    return model


def get_activation(
    model: HookedTransformer,
    prompt: str,
    layer: int,
    component: str = "resid_post",
    return_cpu: bool = True,
) -> torch.Tensor:
    """
    Extract activations from a specific layer and component.

    Args:
        model: HookedTransformer model
        prompt: Input text
        layer: Layer index (0 to n_layers-1)
        component: One of ['resid_pre', 'resid_mid', 'resid_post', 'attn_out', 'mlp_out']
        return_cpu: If True, return on CPU

    Returns:
        Activation tensor of shape [batch, seq_len, d_model]
    """
    cache = {}

    def hook_fn(activation, hook):
        stored = activation.detach()
        if return_cpu:
            stored = stored.cpu()
        cache[hook.name] = stored

    hook_name = f"blocks.{layer}.hook_{component}"
    model.run_with_hooks(
        prompt,
        fwd_hooks=[(hook_name, hook_fn)]
    )

    return cache[hook_name]


def get_attention_patterns(
    model: HookedTransformer,
    prompt: str,
    layer: Optional[int] = None,
    return_cpu: bool = True,
) -> torch.Tensor:
    """
    Get attention patterns for all or specific layer.

    Args:
        model: HookedTransformer model
        prompt: Input text
        layer: Optional layer index. If None, returns all layers.
        return_cpu: If True, return on CPU

    Returns:
        Attention pattern tensor [batch, n_heads, seq_len, seq_len] or
        [batch, n_layers, n_heads, seq_len, seq_len] if layer is None
    """
    _, cache = model.run_with_cache(prompt)

    if layer is not None:
        pattern = cache["pattern", layer]
    else:
        pattern = torch.stack(
            [cache["pattern", l] for l in range(model.cfg.n_layers)],
            dim=1
        )

    if return_cpu:
        pattern = pattern.cpu()

    return pattern


def get_mlp_activations(
    model: HookedTransformer,
    prompt: str,
    layer: int,
    return_cpu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get MLP pre and post activations.

    Args:
        model: HookedTransformer model
        prompt: Input text
        layer: Layer index
        return_cpu: If True, return on CPU

    Returns:
        Tuple of (pre_activation, post_activation)
        pre: [batch, seq_len, d_mlp]
        post: [batch, seq_len, d_mlp]
    """
    cache = {}

    def hook_fn(activation, hook):
        stored = activation.detach()
        if return_cpu:
            stored = stored.cpu()
        cache[hook.name] = stored

    hooks = [
        (f"blocks.{layer}.mlp.hook_pre", hook_fn),
        (f"blocks.{layer}.mlp.hook_post", hook_fn),
    ]

    model.run_with_hooks(prompt, fwd_hooks=hooks)

    return (
        cache[f"blocks.{layer}.mlp.hook_pre"],
        cache[f"blocks.{layer}.mlp.hook_post"],
    )


def cache_all_activations(
    model: HookedTransformer,
    prompt: str,
    components: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Cache all or specific activations from a forward pass.

    Args:
        model: HookedTransformer model
        prompt: Input text
        components: Optional list of components to cache. If None, caches all.

    Returns:
        Tuple of (logits, cache_dict)
    """
    if components is not None:
        # Cache specific components
        names_filter = lambda name: any(comp in name for comp in components)
        logits, cache = model.run_with_cache(prompt, names_filter=names_filter)
    else:
        # Cache everything
        logits, cache = model.run_with_cache(prompt)

    return logits, cache


def get_logit_diff(
    logits: torch.Tensor,
    answer_tokens: List[int],
    per_prompt: bool = False,
) -> torch.Tensor:
    """
    Calculate logit difference between correct and incorrect answers.

    Args:
        logits: Model logits [batch, seq_len, vocab_size]
        answer_tokens: List of [correct_token_id, incorrect_token_id]
        per_prompt: If True, return per-prompt diffs, else return mean

    Returns:
        Logit difference scalar or [batch] tensor
    """
    correct_id, incorrect_id = answer_tokens
    final_logits = logits[:, -1, :]  # [batch, vocab_size]

    correct_logits = final_logits[:, correct_id]
    incorrect_logits = final_logits[:, incorrect_id]

    diff = correct_logits - incorrect_logits

    if per_prompt:
        return diff
    else:
        return diff.mean()


def get_hook_names(model: HookedTransformer, layer: Optional[int] = None) -> List[str]:
    """
    Get all hook names for the model or a specific layer.

    Args:
        model: HookedTransformer model
        layer: Optional layer index

    Returns:
        List of hook names
    """
    all_hook_names = [name for name, _ in model.hook_dict.items()]

    if layer is not None:
        all_hook_names = [name for name in all_hook_names if f"blocks.{layer}" in name]

    return all_hook_names


def save_cache(cache: Dict, path: str) -> None:
    """Save activation cache to disk."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to CPU and save
    cpu_cache = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in cache.items()}
    torch.save(cpu_cache, save_path)

    logger.info(f"Cache saved to {save_path}")


def load_cache(path: str, device: Optional[str] = None) -> Dict:
    """Load activation cache from disk."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = torch.load(path, map_location=device)
    logger.info(f"Cache loaded from {path}")

    return cache
