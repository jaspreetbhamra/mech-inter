"""Circuit discovery utilities: ablation, pruning, and path analysis."""

import torch
from transformer_lens import HookedTransformer
from typing import Optional, List, Tuple, Dict, Callable, Set
import logging
from tqdm.auto import tqdm
import einops

from utils import get_logit_diff

logger = logging.getLogger(__name__)


def zero_ablate_head(
    model: HookedTransformer,
    prompt: str,
    layer: int,
    head: int,
) -> torch.Tensor:
    """
    Zero ablate a specific attention head.

    Args:
        model: HookedTransformer model
        prompt: Input prompt
        layer: Layer index
        head: Head index

    Returns:
        Logits after ablation
    """
    def ablate_hook(activation, hook):
        # activation shape: [batch, seq, n_heads, d_head]
        activation[:, :, head, :] = 0.0
        return activation

    hook_name = f"blocks.{layer}.attn.hook_result"
    ablated_logits = model.run_with_hooks(
        prompt,
        fwd_hooks=[(hook_name, ablate_hook)]
    )

    return ablated_logits


def mean_ablate_head(
    model: HookedTransformer,
    prompt: str,
    layer: int,
    head: int,
    dataset_prompts: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    Mean ablate a specific attention head.

    If dataset_prompts is provided, uses the mean activation from that dataset.
    Otherwise, uses mean over sequence positions.

    Args:
        model: HookedTransformer model
        prompt: Input prompt
        layer: Layer index
        head: Head index
        dataset_prompts: Optional dataset to compute mean from

    Returns:
        Logits after ablation
    """
    if dataset_prompts is not None:
        # Compute mean from dataset
        all_acts = []
        for p in dataset_prompts:
            _, cache = model.run_with_cache(p)
            head_out = cache["result", layer][:, :, head, :]
            all_acts.append(head_out.mean(dim=(0, 1)))  # Mean over batch and seq

        mean_act = torch.stack(all_acts).mean(dim=0)  # [d_head]

        def ablate_hook(activation, hook):
            activation[:, :, head, :] = mean_act.to(activation.device)
            return activation
    else:
        # Use mean over sequence
        def ablate_hook(activation, hook):
            activation[:, :, head, :] = activation[:, :, head, :].mean(dim=1, keepdim=True)
            return activation

    hook_name = f"blocks.{layer}.attn.hook_result"
    ablated_logits = model.run_with_hooks(
        prompt,
        fwd_hooks=[(hook_name, ablate_hook)]
    )

    return ablated_logits


def ablate_all_heads(
    model: HookedTransformer,
    prompt: str,
    answer_tokens: List[int],
    ablation_type: str = "zero",
) -> torch.Tensor:
    """
    Ablate each head individually and measure effect on logit diff.

    Args:
        model: HookedTransformer model
        prompt: Input prompt
        answer_tokens: [correct_token_id, incorrect_token_id]
        ablation_type: "zero" or "mean"

    Returns:
        Tensor of shape [n_layers, n_heads] with ablation effects
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Get baseline
    baseline_logits = model(prompt)
    baseline_logit_diff = get_logit_diff(baseline_logits, answer_tokens)

    results = torch.zeros(n_layers, n_heads)

    ablate_fn = zero_ablate_head if ablation_type == "zero" else mean_ablate_head

    for layer in tqdm(range(n_layers), desc="Ablating heads"):
        for head in range(n_heads):
            ablated_logits = ablate_fn(model, prompt, layer, head)
            ablated_logit_diff = get_logit_diff(ablated_logits, answer_tokens)

            # Effect = baseline - ablated (how much performance drops)
            effect = baseline_logit_diff - ablated_logit_diff
            results[layer, head] = effect.item()

    return results


def ablate_mlp_layer(
    model: HookedTransformer,
    prompt: str,
    layer: int,
    ablation_type: str = "zero",
) -> torch.Tensor:
    """
    Ablate entire MLP layer.

    Args:
        model: HookedTransformer model
        prompt: Input prompt
        layer: Layer index
        ablation_type: "zero" or "mean"

    Returns:
        Logits after ablation
    """
    if ablation_type == "zero":
        def ablate_hook(activation, hook):
            activation[:] = 0.0
            return activation
    else:
        def ablate_hook(activation, hook):
            activation[:] = activation.mean(dim=1, keepdim=True)
            return activation

    hook_name = f"blocks.{layer}.hook_mlp_out"
    ablated_logits = model.run_with_hooks(
        prompt,
        fwd_hooks=[(hook_name, ablate_hook)]
    )

    return ablated_logits


def iterative_pruning(
    model: HookedTransformer,
    prompt: str,
    answer_tokens: List[int],
    threshold: float = 0.01,
    ablation_type: str = "zero",
) -> Set[Tuple[int, int]]:
    """
    Iteratively prune unimportant heads to discover minimal circuit.

    Algorithm:
        1. Ablate each head, measure effect
        2. Keep ablated the head with smallest effect (if below threshold)
        3. Repeat until no more heads can be pruned

    Args:
        model: HookedTransformer model
        prompt: Input prompt
        answer_tokens: [correct_token_id, incorrect_token_id]
        threshold: Maximum effect to allow pruning
        ablation_type: "zero" or "mean"

    Returns:
        Set of (layer, head) tuples representing the important heads
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Start with all heads
    active_heads = {(layer, head) for layer in range(n_layers) for head in range(n_heads)}
    ablated_heads = set()

    # Get baseline
    baseline_logits = model(prompt)
    baseline_logit_diff = get_logit_diff(baseline_logits, answer_tokens)

    logger.info(f"Starting iterative pruning. Baseline logit diff: {baseline_logit_diff:.3f}")

    while True:
        effects = []

        # Try ablating each active head
        for layer, head in active_heads:
            # Create hooks for currently ablated heads + this one
            def make_ablate_hook(h):
                def hook(activation, hook_obj):
                    if ablation_type == "zero":
                        activation[:, :, h, :] = 0.0
                    else:
                        activation[:, :, h, :] = activation[:, :, h, :].mean(dim=1, keepdim=True)
                    return activation
                return hook

            hooks = []
            for abl_layer in range(n_layers):
                # Get heads to ablate in this layer
                heads_to_ablate = [h for l, h in ablated_heads | {(layer, head)} if l == abl_layer]

                if heads_to_ablate:
                    def multi_ablate_hook(activation, hook_obj, heads=heads_to_ablate):
                        for h in heads:
                            if ablation_type == "zero":
                                activation[:, :, h, :] = 0.0
                            else:
                                activation[:, :, h, :] = activation[:, :, h, :].mean(dim=1, keepdim=True)
                        return activation

                    hooks.append((f"blocks.{abl_layer}.attn.hook_result", multi_ablate_hook))

            # Run with ablations
            ablated_logits = model.run_with_hooks(prompt, fwd_hooks=hooks)
            ablated_logit_diff = get_logit_diff(ablated_logits, answer_tokens)

            effect = abs(baseline_logit_diff - ablated_logit_diff)
            effects.append(((layer, head), effect.item()))

        # Find head with smallest effect
        if not effects:
            break

        effects.sort(key=lambda x: x[1])
        (layer, head), min_effect = effects[0]

        # If effect is below threshold, prune this head
        if min_effect < threshold:
            ablated_heads.add((layer, head))
            active_heads.remove((layer, head))
            logger.info(f"Pruned L{layer}H{head} (effect: {min_effect:.4f}). Remaining: {len(active_heads)}")
        else:
            # Can't prune any more
            logger.info(f"Stopping. Minimum effect {min_effect:.4f} exceeds threshold {threshold}")
            break

    logger.info(f"Pruning complete. Important heads: {len(active_heads)}/{n_layers * n_heads}")

    return active_heads


def find_direct_effects(
    model: HookedTransformer,
    prompt: str,
    answer_tokens: List[int],
) -> Dict[str, float]:
    """
    Compute direct effect of each component on logits.

    Direct effect = how much a component's output directly changes the logits
    (via the unembedding), ignoring effects mediated through later layers.

    Args:
        model: HookedTransformer model
        prompt: Input prompt
        answer_tokens: [correct_token_id, incorrect_token_id]

    Returns:
        Dictionary mapping component name -> direct effect on logit diff
    """
    _, cache = model.run_with_cache(prompt)

    # Get unembedding directions for answer tokens
    W_U = model.W_U  # [d_model, vocab_size]
    correct_dir = W_U[:, answer_tokens[0]]  # [d_model]
    incorrect_dir = W_U[:, answer_tokens[1]]  # [d_model]
    logit_diff_dir = correct_dir - incorrect_dir  # [d_model]

    results = {}

    # Attention heads
    for layer in range(model.cfg.n_layers):
        head_outputs = cache["result", layer]  # [batch, seq, n_heads, d_head]

        # Project to residual stream
        W_O = model.W_O[layer]  # [n_heads, d_head, d_model]
        head_residual = einops.einsum(
            head_outputs,
            W_O,
            "batch seq head d_head, head d_head d_model -> batch seq head d_model"
        )

        # Direct effect on logit diff
        for head in range(model.cfg.n_heads):
            head_output = head_residual[0, -1, head, :]  # Final token, [d_model]
            direct_effect = (head_output @ logit_diff_dir).item()
            results[f"L{layer}H{head}"] = direct_effect

    # MLPs
    for layer in range(model.cfg.n_layers):
        mlp_output = cache["mlp_out", layer][0, -1, :]  # [d_model]
        direct_effect = (mlp_output @ logit_diff_dir).item()
        results[f"L{layer}_MLP"] = direct_effect

    return results


def attention_knockout(
    model: HookedTransformer,
    prompt: str,
    layer: int,
    head: int,
    answer_tokens: List[int],
) -> Dict[int, float]:
    """
    Knock out attention from specific head to each position, measure effect.

    Args:
        model: HookedTransformer model
        prompt: Input prompt
        layer: Layer index
        head: Head index
        answer_tokens: [correct_token_id, incorrect_token_id]

    Returns:
        Dictionary mapping position -> effect of knocking out attention to that position
    """
    tokens = model.to_tokens(prompt)
    seq_len = tokens.shape[1]

    baseline_logits = model(prompt)
    baseline_logit_diff = get_logit_diff(baseline_logits, answer_tokens)

    results = {}

    for pos in range(seq_len):
        def knockout_hook(pattern, hook):
            # pattern shape: [batch, head, query_pos, key_pos]
            # Zero out attention from all positions to this key position
            pattern[:, head, :, pos] = 0.0
            # Renormalize
            pattern[:, head, :, :] = pattern[:, head, :, :] / pattern[:, head, :, :].sum(dim=-1, keepdim=True)
            return pattern

        hook_name = f"blocks.{layer}.attn.hook_pattern"
        knockout_logits = model.run_with_hooks(
            prompt,
            fwd_hooks=[(hook_name, knockout_hook)]
        )

        knockout_logit_diff = get_logit_diff(knockout_logits, answer_tokens)
        effect = baseline_logit_diff - knockout_logit_diff
        results[pos] = effect.item()

    return results
