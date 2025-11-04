"""Demonstration of activation extraction and visualization for fact checking.

This script shows how to:
1. Load a model (GPT-2 Medium or Llama 3.2 1B)
2. Create/load a fact dataset
3. Extract activations for true and false facts
4. Visualize activation patterns using heatmaps and dimensionality reduction

Usage:
    python experiments/activation_analysis_demo.py --model gpt2-medium
    python experiments/activation_analysis_demo.py --model meta-llama/Llama-3.2-1B
"""

import argparse
import logging
from pathlib import Path

import torch

from src.utils import setup_logging, set_seed, load_model
from src.activation_extraction import ActivationExtractor, ActivationConfig
from src.fact_dataset import FactDataset, load_or_create_dataset
from src.visualization import (
    plot_activation_magnitude_heatmap,
    plot_activation_comparison,
    plot_pca_activations,
    plot_activation_space_comparison,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze model activations for fact checking"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-medium",
        choices=["gpt2-medium", "meta-llama/Llama-3.2-1B"],
        help="Model to use for analysis",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/facts.json",
        help="Path to fact dataset JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/activation_analysis",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--component",
        type=str,
        default="resid_post",
        choices=["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"],
        help="Component to extract activations from",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Specific layer to visualize (default: visualize all layers)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--save-html",
        action="store_true",
        help="Save visualizations as interactive HTML files",
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Setup
    setup_logging(level=logging.INFO)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Activation Analysis Demo")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Load Model
    # -------------------------------------------------------------------------
    logger.info(f"\n[Step 1/5] Loading model: {args.model}")

    model = load_model(args.model)

    logger.info(f"Model loaded successfully!")
    logger.info(f"  - Layers: {model.cfg.n_layers}")
    logger.info(f"  - Heads: {model.cfg.n_heads}")
    logger.info(f"  - d_model: {model.cfg.d_model}")

    # -------------------------------------------------------------------------
    # 2. Load or Create Fact Dataset
    # -------------------------------------------------------------------------
    logger.info(f"\n[Step 2/5] Loading fact dataset from: {args.dataset}")

    dataset = load_or_create_dataset(args.dataset, create_if_missing=True)

    logger.info(f"Dataset loaded: {dataset}")

    # Show some examples
    logger.info("\nExample facts:")
    for i in range(min(3, len(dataset))):
        fact = dataset[i]
        prompt = fact.to_prompt()
        logger.info(f"  [{i}] {'✓' if fact.is_true else '✗'} {prompt}")

    # -------------------------------------------------------------------------
    # 3. Extract Activations
    # -------------------------------------------------------------------------
    logger.info(f"\n[Step 3/5] Extracting activations for component: {args.component}")

    # Configure extraction
    config = ActivationConfig(
        components=[args.component],
        aggregate_positions=True,  # Average over sequence length
        return_cpu=True,
        cache_path=output_dir / f"activations_{args.model.replace('/', '_')}.pt",
    )

    extractor = ActivationExtractor(model, config)

    # Get prompts for true and false facts
    true_dataset = dataset.filter(is_true=True)
    false_dataset = dataset.filter(is_true=False)

    true_prompts = true_dataset.to_prompts()
    false_prompts = false_dataset.to_prompts()

    logger.info(f"  - True facts: {len(true_prompts)}")
    logger.info(f"  - False facts: {len(false_prompts)}")

    # Extract activations
    logger.info("Extracting activations for true facts...")
    true_activations = extractor.extract(true_prompts)

    logger.info("Extracting activations for false facts...")
    false_activations = extractor.extract(false_prompts)

    # Get the component activations
    true_acts = true_activations[args.component]  # [n_true, n_layers, d_model]
    false_acts = false_activations[args.component]  # [n_false, n_layers, d_model]

    logger.info(f"Activations extracted!")
    logger.info(f"  - True shape: {true_acts.shape}")
    logger.info(f"  - False shape: {false_acts.shape}")

    # -------------------------------------------------------------------------
    # 4. Compute Activation Statistics
    # -------------------------------------------------------------------------
    logger.info(f"\n[Step 4/5] Computing activation statistics")

    true_stats = extractor.get_activation_stats({args.component: true_acts})
    false_stats = extractor.get_activation_stats({args.component: false_acts})

    logger.info("\nTrue facts statistics:")
    for stat, value in true_stats[args.component].items():
        logger.info(f"  {stat}: {value:.4f}")

    logger.info("\nFalse facts statistics:")
    for stat, value in false_stats[args.component].items():
        logger.info(f"  {stat}: {value:.4f}")

    # -------------------------------------------------------------------------
    # 5. Create Visualizations
    # -------------------------------------------------------------------------
    logger.info(f"\n[Step 5/5] Creating visualizations")

    # 5.1 Heatmap of activation magnitudes
    logger.info("Creating activation magnitude heatmap...")

    all_acts = torch.cat([true_acts, false_acts], dim=0)
    labels = (
        [f"True: {p[:30]}..." for p in true_prompts] +
        [f"False: {p[:30]}..." for p in false_prompts]
    )

    fig_heatmap = plot_activation_magnitude_heatmap(
        all_acts,
        labels=labels,
        title=f"Activation Magnitudes - {args.component}",
    )

    if args.save_html:
        fig_heatmap.write_html(output_dir / "heatmap.html")
        logger.info(f"  Saved: {output_dir / 'heatmap.html'}")
    else:
        fig_heatmap.show()

    # 5.2 Compare true vs false activations across layers
    logger.info("Creating activation comparison plot...")

    fig_comparison = plot_activation_comparison(
        true_acts,
        false_acts,
        layer_idx=args.layer,
        title=f"True vs False Facts - {args.component}",
    )

    if args.save_html:
        fig_comparison.write_html(output_dir / "comparison.html")
        logger.info(f"  Saved: {output_dir / 'comparison.html'}")
    else:
        fig_comparison.show()

    # 5.3 PCA visualization
    logger.info("Creating PCA visualization...")

    # Select layer for PCA (use middle layer if not specified)
    pca_layer = args.layer if args.layer is not None else model.cfg.n_layers // 2

    fig_pca = plot_activation_space_comparison(
        true_acts,
        false_acts,
        method="pca",
        layer_idx=pca_layer,
        title=f"PCA - Layer {pca_layer} - {args.component}",
    )

    if args.save_html:
        fig_pca.write_html(output_dir / "pca.html")
        logger.info(f"  Saved: {output_dir / 'pca.html'}")
    else:
        fig_pca.show()

    # 5.4 t-SNE visualization (optional, takes longer)
    logger.info("Creating t-SNE visualization...")

    fig_tsne = plot_activation_space_comparison(
        true_acts,
        false_acts,
        method="tsne",
        layer_idx=pca_layer,
        title=f"t-SNE - Layer {pca_layer} - {args.component}",
    )

    if args.save_html:
        fig_tsne.write_html(output_dir / "tsne.html")
        logger.info(f"  Saved: {output_dir / 'tsne.html'}")
    else:
        fig_tsne.show()

    # 5.5 UMAP visualization (if available)
    try:
        logger.info("Creating UMAP visualization...")

        fig_umap = plot_activation_space_comparison(
            true_acts,
            false_acts,
            method="umap",
            layer_idx=pca_layer,
            title=f"UMAP - Layer {pca_layer} - {args.component}",
        )

        if args.save_html:
            fig_umap.write_html(output_dir / "umap.html")
            logger.info(f"  Saved: {output_dir / 'umap.html'}")
        else:
            fig_umap.show()

    except ImportError:
        logger.warning("UMAP not available. Install with: pip install umap-learn")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("Analysis Complete!")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Component: {args.component}")
    logger.info(f"True facts: {len(true_prompts)}")
    logger.info(f"False facts: {len(false_prompts)}")

    if args.save_html:
        logger.info(f"\nVisualizations saved to: {output_dir}")
        logger.info("  - heatmap.html")
        logger.info("  - comparison.html")
        logger.info("  - pca.html")
        logger.info("  - tsne.html")
        logger.info("  - umap.html (if available)")

    logger.info("\nKey Findings:")
    logger.info(f"  True facts - Mean activation L2 norm: {true_stats[args.component]['l2_norm']:.4f}")
    logger.info(f"  False facts - Mean activation L2 norm: {false_stats[args.component]['l2_norm']:.4f}")

    diff = true_stats[args.component]['l2_norm'] - false_stats[args.component]['l2_norm']
    logger.info(f"  Difference: {diff:.4f} ({diff / true_stats[args.component]['l2_norm'] * 100:.2f}%)")


if __name__ == "__main__":
    main()
