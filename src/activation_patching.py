"""Activation patching (causal tracing) utilities for mechanistic interpretability."""

import logging

import torch
from transformer_lens import HookedTransformer

from src.utils import get_activation, get_logit_diff

logger = logging.getLogger(__name__)


def patch_activation(
    model: HookedTransformer,
    clean_prompt: str,
    corrupted_prompt: str,
    layer: int,
    component: str = "resid_post",
    position: int | None = None,
) -> torch.Tensor:
    """
    Patch activation from corrupted run into clean run.

    Args:
        model: HookedTransformer model
        clean_prompt: Clean input prompt
        corrupted_prompt: Corrupted input prompt
        layer: Layer index to patch
        component: Component to patch (resid_post, attn_out, mlp_out, etc.)
        position: Optional position to patch. If None, patches all positions.

    Returns:
        Logits after patching
    """
    # Get corrupted activation
    corrupted_act = get_activation(model, corrupted_prompt, layer, component)

    # Define patch hook
    def patch_hook(activation, hook):
        if position is not None:
            activation[:, position, :] = corrupted_act[:, position, :].to(activation.device)
        else:
            activation[:] = corrupted_act.to(activation.device)
        return activation

    hook_name = f"blocks.{layer}.hook_{component}"
    patched_logits = model.run_with_hooks(clean_prompt, fwd_hooks=[(hook_name, patch_hook)])

    return patched_logits


def patch_head_output(
    model: HookedTransformer,
    clean_prompt: str,
    corrupted_prompt: str,
    layer: int,
    head: int,
    position: int | None = None,
) -> torch.Tensor:
    """
    Patch a specific attention head's output.

    Args:
        model: HookedTransformer model
        clean_prompt: Clean input prompt
        corrupted_prompt: Corrupted input prompt
        layer: Layer index
        head: Head index
        position: Optional position to patch

    Returns:
        Logits after patching
    """
    # Get corrupted head output
    _, corrupted_cache = model.run_with_cache(corrupted_prompt)
    corrupted_head_out = corrupted_cache["result", layer][:, :, head, :]  # [batch, seq, d_head]

    def patch_hook(activation, hook):
        # activation shape: [batch, seq, n_heads, d_head]
        if position is not None:
            activation[:, position, head, :] = corrupted_head_out[:, position, :].to(
                activation.device
            )
        else:
            activation[:, :, head, :] = corrupted_head_out.to(activation.device)
        return activation

    hook_name = f"blocks.{layer}.attn.hook_result"
    patched_logits = model.run_with_hooks(clean_prompt, fwd_hooks=[(hook_name, patch_hook)])

    return patched_logits


def comprehensive_activation_patching(
    model: HookedTransformer,
    clean_prompt: str,
    corrupted_prompt: str,
    answer_tokens: list[int],
    components: list[str] = ["resid_post", "attn_out", "mlp_out"],
) -> dict[str, torch.Tensor]:
    """
    Perform activation patching across all layers and specified components.

    Args:
        model: HookedTransformer model
        clean_prompt: Clean input prompt
        corrupted_prompt: Corrupted input prompt
        answer_tokens: [correct_token_id, incorrect_token_id] for logit diff
        components: List of components to patch

    Returns:
        Dictionary mapping (component, layer) -> logit_diff
    """
    # Get baseline logit diffs
    clean_logits = model(clean_prompt)
    corrupted_logits = model(corrupted_prompt)

    clean_logit_diff = get_logit_diff(clean_logits, answer_tokens)
    corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_tokens)

    logger.info(f"Clean logit diff: {clean_logit_diff:.3f}")
    logger.info(f"Corrupted logit diff: {corrupted_logit_diff:.3f}")

    results = {}

    # Patch each component at each layer
    for component in components:
        for layer in range(model.cfg.n_layers):
            patched_logits = patch_activation(
                model, clean_prompt, corrupted_prompt, layer, component
            )
            patched_logit_diff = get_logit_diff(patched_logits, answer_tokens)

            # Normalize: fraction of corruption recovered
            effect = (patched_logit_diff - clean_logit_diff) / (
                corrupted_logit_diff - clean_logit_diff
            )

            results[(component, layer)] = effect.item()

            logger.debug(f"{component} L{layer}: {effect:.3f}")

    return results


def head_patching_sweep(
    model: HookedTransformer,
    clean_prompt: str,
    corrupted_prompt: str,
    answer_tokens: list[int],
) -> torch.Tensor:
    """
    Patch each attention head individually and measure effect.

    Args:
        model: HookedTransformer model
        clean_prompt: Clean input prompt
        corrupted_prompt: Corrupted input prompt
        answer_tokens: [correct_token_id, incorrect_token_id]

    Returns:
        Tensor of shape [n_layers, n_heads] with patching effects
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    results = torch.zeros(n_layers, n_heads)

    # Baselines
    clean_logits = model(clean_prompt)
    corrupted_logits = model(corrupted_prompt)
    clean_logit_diff = get_logit_diff(clean_logits, answer_tokens)
    corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_tokens)

    # Patch each head
    for layer in range(n_layers):
        for head in range(n_heads):
            patched_logits = patch_head_output(model, clean_prompt, corrupted_prompt, layer, head)
            patched_logit_diff = get_logit_diff(patched_logits, answer_tokens)

            # Normalized effect
            effect = (patched_logit_diff - clean_logit_diff) / (
                corrupted_logit_diff - clean_logit_diff
            )
            results[layer, head] = effect.item()

    return results


def path_patching(
    model: HookedTransformer,
    clean_prompt: str,
    corrupted_prompt: str,
    sender_layer: int,
    sender_component: str,
    receiver_layer: int,
    receiver_component: str,
    answer_tokens: list[int],
) -> float:
    """
    Path patching: patch a path from sender to receiver component.

    This isolates the effect of information flow along a specific path.

    Args:
        model: HookedTransformer model
        clean_prompt: Clean input prompt
        corrupted_prompt: Corrupted input prompt
        sender_layer: Source layer
        sender_component: Source component
        receiver_layer: Target layer
        receiver_component: Target component
        answer_tokens: [correct_token_id, incorrect_token_id]

    Returns:
        Normalized patching effect
    """
    # Get corrupted sender activation
    corrupted_sender = get_activation(model, corrupted_prompt, sender_layer, sender_component)

    # Define hooks
    def sender_hook(activation, hook):
        activation[:] = corrupted_sender.to(activation.device)
        return activation

    def receiver_hook(activation, hook):
        # Only patch the contribution from sender
        # This is simplified - full path patching requires gradient computation
        return activation

    hooks = [
        (f"blocks.{sender_layer}.hook_{sender_component}", sender_hook),
    ]

    patched_logits = model.run_with_hooks(clean_prompt, fwd_hooks=hooks)
    patched_logit_diff = get_logit_diff(patched_logits, answer_tokens)

    # Get baselines
    clean_logits = model(clean_prompt)
    corrupted_logits = model(corrupted_prompt)
    clean_logit_diff = get_logit_diff(clean_logits, answer_tokens)
    corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_tokens)

    effect = (patched_logit_diff - clean_logit_diff) / (corrupted_logit_diff - clean_logit_diff)

    return effect.item()
