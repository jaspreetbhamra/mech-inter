"""Comprehensive activation extraction for mechanistic interpretability.

This module provides functions to extract and organize activations from
TransformerLens models across multiple prompts and layers.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)


@dataclass
class ActivationConfig:
    """Configuration for activation extraction.

    Attributes:
        components: List of components to extract. Options:
            - 'resid_pre': Residual stream before attention+MLP
            - 'resid_mid': Residual stream after attention, before MLP
            - 'resid_post': Residual stream after attention+MLP
            - 'attn_out': Attention output (before added to residual)
            - 'mlp_out': MLP output (before added to residual)
            - 'attn_pre': Attention input (queries, keys, values)
            - 'mlp_pre': MLP pre-activation
            - 'mlp_post': MLP post-activation
        layers: Specific layers to extract from. If None, extracts all layers.
        positions: Specific token positions to extract. If None, extracts all.
        aggregate_positions: If True, average across sequence positions.
        return_cpu: If True, move all activations to CPU.
        cache_path: Optional path to cache activations to disk.
    """

    components: list[str] = field(
        default_factory=lambda: ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"]
    )
    layers: list[int] | None = None
    positions: list[int] | None = None
    aggregate_positions: bool = False
    return_cpu: bool = True
    cache_path: str | None = None


class ActivationExtractor:
    """Extracts and organizes activations from TransformerLens models.

    Example:
        >>> model = HookedTransformer.from_pretrained('gpt2-medium')
        >>> extractor = ActivationExtractor(model)
        >>> prompts = ["The Eiffel Tower is in Paris", "Paris is in France"]
        >>> activations = extractor.extract(prompts)
        >>> print(activations.keys())  # dict_keys(['resid_pre', 'resid_mid', ...])
    """

    def __init__(
        self,
        model: HookedTransformer,
        config: ActivationConfig | None = None,
    ):
        """Initialize the activation extractor.

        Args:
            model: HookedTransformer model to extract activations from.
            config: Configuration for extraction. Uses defaults if None.
        """
        self.model = model
        self.config = config or ActivationConfig()

        # Validate configuration
        self._validate_config()

        # Set up layers to extract
        if self.config.layers is None:
            self.layers = list(range(self.model.cfg.n_layers))
        else:
            self.layers = self.config.layers

        logger.info(
            f"Initialized ActivationExtractor for {model.cfg.model_name} "
            f"with {len(self.config.components)} components across {len(self.layers)} layers"
        )

    def _validate_config(self) -> None:
        """Validate the configuration settings."""
        valid_components = {
            "resid_pre",
            "resid_mid",
            "resid_post",
            "attn_out",
            "mlp_out",
            "attn_pre",
            "mlp_pre",
            "mlp_post",
        }
        for comp in self.config.components:
            if comp not in valid_components:
                raise ValueError(f"Invalid component '{comp}'. Must be one of {valid_components}")

    def extract(
        self,
        prompts: str | list[str],
        show_progress: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Extract activations for a list of prompts.

        Args:
            prompts: Single prompt or list of prompts to extract activations from.
            show_progress: Whether to show progress bar.

        Returns:
            Dictionary mapping component names to activation tensors.
            Shape of tensors: [n_prompts, n_layers, seq_len, d_model]
            (or [n_prompts, n_layers, d_model] if aggregate_positions=True)
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        logger.info(f"Extracting activations for {len(prompts)} prompts...")

        # Check cache first
        if self.config.cache_path is not None:
            cache_file = Path(self.config.cache_path)
            if cache_file.exists():
                logger.info(f"Loading cached activations from {cache_file}")
                return torch.load(cache_file)

        # Initialize storage for each component
        all_activations: dict[str, list[torch.Tensor]] = {
            comp: [] for comp in self.config.components
        }

        # Extract activations for each prompt
        iterator = tqdm(prompts, desc="Extracting") if show_progress else prompts
        for prompt in iterator:
            prompt_acts = self._extract_single(prompt)

            for comp, acts in prompt_acts.items():
                # Aggregate positions now if requested (to handle variable sequence lengths)
                if self.config.aggregate_positions:
                    # Average across sequence: [n_layers, d_model]
                    acts = acts.mean(dim=1)
                all_activations[comp].append(acts)

        # Stack activations across prompts
        result = {}
        for comp, acts_list in all_activations.items():
            if self.config.aggregate_positions:
                # Stack: [n_prompts, n_layers, d_model]
                result[comp] = torch.stack(acts_list, dim=0)
            else:
                # Variable sequence lengths - can't stack directly
                # Need to pad or return list
                # For now, pad to max length
                max_len = max(a.shape[1] for a in acts_list)
                padded_acts = []
                for acts in acts_list:
                    if acts.shape[1] < max_len:
                        # Pad with zeros: [n_layers, seq_len, d_model]
                        pad_size = max_len - acts.shape[1]
                        padding = torch.zeros(
                            acts.shape[0], pad_size, acts.shape[2],
                            dtype=acts.dtype, device=acts.device
                        )
                        acts = torch.cat([acts, padding], dim=1)
                    padded_acts.append(acts)
                # Stack: [n_prompts, n_layers, seq_len, d_model]
                result[comp] = torch.stack(padded_acts, dim=0)

        # Cache results if requested
        if self.config.cache_path is not None:
            cache_file = Path(self.config.cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result, cache_file)
            logger.info(f"Cached activations to {cache_file}")

        return result

    def _extract_single(self, prompt: str) -> dict[str, torch.Tensor]:
        """Extract activations for a single prompt.

        Returns:
            Dictionary mapping component names to tensors of shape [n_layers, seq_len, d_model]
        """
        # Run forward pass with cache
        _, cache = self.model.run_with_cache(prompt)

        activations = {}

        for component in self.config.components:
            layer_acts = []

            for layer in self.layers:
                # Get activation based on component type
                if component == "resid_pre":
                    act = cache["resid_pre", layer]
                elif component == "resid_mid":
                    act = cache["resid_mid", layer]
                elif component == "resid_post":
                    act = cache["resid_post", layer]
                elif component == "attn_out":
                    act = cache["attn_out", layer]
                elif component == "mlp_out":
                    act = cache["mlp_out", layer]
                elif component == "attn_pre":
                    # This is more complex - returns Q, K, V
                    act = cache["attn_in", layer]
                elif component == "mlp_pre":
                    act = cache["mlp_pre", layer] if "mlp_pre" in cache else None
                elif component == "mlp_post":
                    act = cache["mlp_post", layer] if "mlp_post" in cache else None
                else:
                    raise ValueError(f"Unknown component: {component}")

                if act is None:
                    logger.warning(f"Component {component} not found in cache for layer {layer}")
                    continue

                # Extract specific positions if requested
                if self.config.positions is not None:
                    act = act[:, self.config.positions, :]

                # Move to CPU if requested
                if self.config.return_cpu:
                    act = act.cpu()

                # Remove batch dimension (batch size is always 1 for single prompt)
                act = act[0]  # [seq_len, d_model]

                layer_acts.append(act)

            # Stack across layers: [n_layers, seq_len, d_model]
            if layer_acts:
                activations[component] = torch.stack(layer_acts, dim=0)

        return activations

    def extract_by_layer(
        self,
        prompts: str | list[str],
        layer: int,
    ) -> dict[str, torch.Tensor]:
        """Extract activations for a specific layer only.

        Args:
            prompts: Single prompt or list of prompts.
            layer: Layer index to extract from.

        Returns:
            Dictionary mapping component names to tensors of shape [n_prompts, seq_len, d_model]
        """
        original_layers = self.config.layers
        self.config.layers = [layer]

        result = self.extract(prompts, show_progress=False)

        # Remove layer dimension
        result = {comp: acts[:, 0, ...] for comp, acts in result.items()}

        # Restore original layer config
        self.config.layers = original_layers

        return result

    def extract_final_token(
        self,
        prompts: str | list[str],
    ) -> dict[str, torch.Tensor]:
        """Extract activations at the final token position only.

        Useful for next-token prediction tasks.

        Args:
            prompts: Single prompt or list of prompts.

        Returns:
            Dictionary mapping component names to tensors of shape [n_prompts, n_layers, d_model]
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        all_acts = {comp: [] for comp in self.config.components}

        for prompt in prompts:
            # Get all activations
            prompt_acts = self._extract_single(prompt)

            # Extract final token position
            for comp, acts in prompt_acts.items():
                # acts: [n_layers, seq_len, d_model]
                final_token_acts = acts[:, -1, :]  # [n_layers, d_model]
                all_acts[comp].append(final_token_acts)

        # Stack across prompts
        return {comp: torch.stack(acts, dim=0) for comp, acts in all_acts.items()}

    def get_activation_stats(
        self,
        activations: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, float]]:
        """Compute statistics about extracted activations.

        Args:
            activations: Dictionary of activation tensors.

        Returns:
            Dictionary mapping component -> statistic name -> value
        """
        stats = {}

        for comp, acts in activations.items():
            stats[comp] = {
                "mean": acts.mean().item(),
                "std": acts.std().item(),
                "min": acts.min().item(),
                "max": acts.max().item(),
                "l2_norm": acts.norm(p=2).item() / acts.numel() ** 0.5,
                "sparsity": (acts.abs() < 1e-6).float().mean().item(),
            }

        return stats


def extract_activations_batch(
    model: HookedTransformer,
    prompts: list[str],
    components: list[str] | None = None,
    batch_size: int = 8,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Convenience function to extract activations in batches.

    Args:
        model: HookedTransformer model.
        prompts: List of prompts to process.
        components: Components to extract. Uses defaults if None.
        batch_size: Number of prompts to process at once.
        **kwargs: Additional arguments for ActivationConfig.

    Returns:
        Dictionary of activation tensors.
    """
    if components is not None:
        kwargs["components"] = components

    config = ActivationConfig(**kwargs)
    extractor = ActivationExtractor(model, config)

    # Process in batches to avoid memory issues
    all_results = {comp: [] for comp in config.components}

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        batch_acts = extractor.extract(batch, show_progress=False)

        for comp, acts in batch_acts.items():
            all_results[comp].append(acts)

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate batches
    return {comp: torch.cat(acts, dim=0) for comp, acts in all_results.items()}
