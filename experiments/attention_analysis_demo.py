"""Demonstration of attention analysis for identifying factual recall heads.

This script shows how to:
1. Extract attention patterns from all heads and layers
2. Identify subject tokens in factual statements
3. Find "factual recall heads" using statistical testing
4. Visualize attention patterns with interactive plots
5. Compare attention for true vs false facts

Usage:
    python experiments/attention_analysis_demo.py --model gpt2-medium
    python experiments/attention_analysis_demo.py --model meta-llama/Llama-3.2-1B --top-k 15
"""

import argparse
import logging
from pathlib import Path

import torch

from src.utils import setup_logging, set_seed, load_model
from src.attention_analysis import AttentionAnalyzer, compute_bonferroni_correction
from src.fact_dataset import FactDataset, load_or_create_dataset
from src.visualization import (
    plot_factual_recall_heads,
    plot_attention_to_subject,
    plot_attention_comparison_interactive,
    plot_head_scores_distribution,
    plot_aggregated_attention_flow,
    plot_top_heads_comparison,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze attention patterns for factual recall"
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
        default="outputs/attention_analysis",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top heads to analyze in detail",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help="P-value threshold for significance",
    )
    parser.add_argument(
        "--min-effect-size",
        type=float,
        default=0.05,
        help="Minimum effect size to consider significant",
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
    parser.add_argument(
        "--use-bonferroni",
        action="store_true",
        help="Apply Bonferroni correction for multiple comparisons",
    )
    return parser.parse_args()


def extract_subject_from_fact(fact) -> list:
    """Extract subject tokens from a fact."""
    # Simple heuristic: split subject by spaces and use each word
    return fact.subject.split()


def main():
    """Main execution function."""
    args = parse_args()

    # Setup
    setup_logging(level=logging.INFO)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Attention Analysis Demo - Factual Recall Heads")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Load Model
    # -------------------------------------------------------------------------
    logger.info(f"\n[Step 1/6] Loading model: {args.model}")

    model = load_model(args.model)

    logger.info(f"Model loaded successfully!")
    logger.info(f"  - Layers: {model.cfg.n_layers}")
    logger.info(f"  - Heads: {model.cfg.n_heads}")
    logger.info(f"  - Total heads: {model.cfg.n_layers * model.cfg.n_heads}")

    # -------------------------------------------------------------------------
    # 2. Load Fact Dataset
    # -------------------------------------------------------------------------
    logger.info(f"\n[Step 2/6] Loading fact dataset from: {args.dataset}")

    dataset = load_or_create_dataset(args.dataset, create_if_missing=True)

    logger.info(f"Dataset loaded: {dataset}")

    # Get true and false fact subsets
    true_dataset = dataset.filter(is_true=True)
    false_dataset = dataset.filter(is_true=False)

    # -------------------------------------------------------------------------
    # 3. Extract Attention Patterns
    # -------------------------------------------------------------------------
    logger.info(f"\n[Step 3/6] Extracting attention patterns")

    analyzer = AttentionAnalyzer(model)

    # Extract patterns for true facts
    logger.info(f"Extracting attention for {len(true_dataset)} true facts...")
    true_prompts = true_dataset.to_prompts()
    true_subjects = [extract_subject_from_fact(f) for f in true_dataset.facts]

    true_patterns = analyzer.extract_attention_patterns(
        true_prompts,
        identify_subject=True,
        subject_markers=None,  # Will be set individually below
    )

    # Set subject positions manually based on fact subjects
    for i, pattern in enumerate(true_patterns):
        pattern.subject_positions = analyzer._find_token_positions(
            pattern.tokens, true_subjects[i]
        )

    # Extract patterns for false facts
    logger.info(f"Extracting attention for {len(false_dataset)} false facts...")
    false_prompts = false_dataset.to_prompts()
    false_subjects = [extract_subject_from_fact(f) for f in false_dataset.facts]

    false_patterns = analyzer.extract_attention_patterns(
        false_prompts,
        identify_subject=True,
        subject_markers=None,
    )

    for i, pattern in enumerate(false_patterns):
        pattern.subject_positions = analyzer._find_token_positions(
            pattern.tokens, false_subjects[i]
        )

    logger.info(f"Attention extraction complete!")

    # Show example
    logger.info("\nExample attention pattern:")
    example = true_patterns[0]
    logger.info(f"  Prompt: {example.prompt}")
    logger.info(f"  Tokens: {example.tokens}")
    logger.info(f"  Subject positions: {example.subject_positions}")
    logger.info(f"  Prediction position: {example.prediction_position}")

    # -------------------------------------------------------------------------
    # 4. Identify Factual Recall Heads
    # -------------------------------------------------------------------------
    logger.info(f"\n[Step 4/6] Identifying factual recall heads")
    logger.info(f"  P-value threshold: {args.p_threshold}")
    logger.info(f"  Minimum effect size: {args.min_effect_size}")

    results = analyzer.identify_factual_recall_heads(
        true_patterns,
        false_patterns,
        threshold=args.p_threshold,
        min_effect_size=args.min_effect_size,
    )

    significant_heads = results['significant_heads']
    n_significant = len(significant_heads)

    logger.info(f"\nFound {n_significant} significant heads!")

    if args.use_bonferroni:
        logger.info("\nApplying Bonferroni correction...")
        sig_mask, corrected_alpha = compute_bonferroni_correction(
            results['p_values'], alpha=args.p_threshold
        )
        n_corrected = sig_mask.sum().item()
        logger.info(f"  Corrected alpha: {corrected_alpha:.2e}")
        logger.info(f"  Significant heads after correction: {n_corrected}")

    # Display top heads
    logger.info(f"\nTop {args.top_k} factual recall heads:")
    for i, head in enumerate(significant_heads[:args.top_k]):
        logger.info(
            f"  {i+1}. Layer {head.layer} Head {head.head}: "
            f"effect={head.score:.4f}, p={head.p_value:.2e}"
        )

    # -------------------------------------------------------------------------
    # 5. Statistical Summary
    # -------------------------------------------------------------------------
    logger.info(f"\n[Step 5/6] Computing statistical summary")

    true_scores_mean = results['true_scores']
    false_scores_mean = results['false_scores']
    effect_sizes = results['effect_sizes']

    logger.info(f"\nOverall statistics:")
    logger.info(f"  Mean attention (true facts): {true_scores_mean.mean():.4f}")
    logger.info(f"  Mean attention (false facts): {false_scores_mean.mean():.4f}")
    logger.info(f"  Mean effect size: {effect_sizes.mean():.4f}")
    logger.info(f"  Max effect size: {effect_sizes.max():.4f}")
    logger.info(f"  Min effect size: {effect_sizes.min():.4f}")

    # Find heads with largest positive and negative effects
    flat_effects = effect_sizes.flatten()
    max_idx = flat_effects.argmax()
    min_idx = flat_effects.argmin()

    max_layer = max_idx // model.cfg.n_heads
    max_head = max_idx % model.cfg.n_heads
    min_layer = min_idx // model.cfg.n_heads
    min_head = min_idx % model.cfg.n_heads

    logger.info(f"\nLargest positive effect:")
    logger.info(f"  Layer {max_layer} Head {max_head}: {flat_effects[max_idx]:.4f}")
    logger.info(f"Largest negative effect:")
    logger.info(f"  Layer {min_layer} Head {min_head}: {flat_effects[min_idx]:.4f}")

    # -------------------------------------------------------------------------
    # 6. Create Visualizations
    # -------------------------------------------------------------------------
    logger.info(f"\n[Step 6/6] Creating visualizations")

    # 6.1 Overview heatmap of all heads
    logger.info("Creating factual recall heads heatmap...")

    fig_overview = plot_factual_recall_heads(
        results,
        top_k=args.top_k,
        title=f"Factual Recall Heads - {args.model}",
    )

    if args.save_html:
        fig_overview.write_html(output_dir / "recall_heads_overview.html")
        logger.info(f"  Saved: {output_dir / 'recall_heads_overview.html'}")
    else:
        fig_overview.show()

    # 6.2 Top heads comparison
    if significant_heads:
        logger.info("Creating top heads comparison...")

        fig_top_heads = plot_top_heads_comparison(
            significant_heads,
            true_scores_mean,
            false_scores_mean,
            top_k=min(args.top_k, len(significant_heads)),
        )

        if args.save_html:
            fig_top_heads.write_html(output_dir / "top_heads_comparison.html")
            logger.info(f"  Saved: {output_dir / 'top_heads_comparison.html'}")
        else:
            fig_top_heads.show()

    # 6.3 Detailed attention pattern for top head
    if significant_heads:
        logger.info("Creating detailed attention pattern for top head...")

        top_head = significant_heads[0]
        layer, head = top_head.layer, top_head.head

        # Show pattern for first true fact
        fig_attn = plot_attention_to_subject(
            true_patterns[0],
            layer,
            head,
            title=f"Attention Pattern - Top Head L{layer}H{head}",
        )

        if args.save_html:
            fig_attn.write_html(output_dir / "top_head_attention.html")
            logger.info(f"  Saved: {output_dir / 'top_head_attention.html'}")
        else:
            fig_attn.show()

    # 6.4 True vs False comparison for top head
    if significant_heads and len(true_patterns) > 0 and len(false_patterns) > 0:
        logger.info("Creating true vs false attention comparison...")

        fig_comparison = plot_attention_comparison_interactive(
            true_patterns[0],
            false_patterns[0],
            layer,
            head,
        )

        if args.save_html:
            fig_comparison.write_html(output_dir / "attention_comparison.html")
            logger.info(f"  Saved: {output_dir / 'attention_comparison.html'}")
        else:
            fig_comparison.show()

    # 6.5 Score distribution for top head
    if significant_heads:
        logger.info("Creating score distribution for top head...")

        # Need to compute all scores
        true_scores_all = results['true_scores']
        false_scores_all = results['false_scores']

        # Get scores for all samples
        from src.attention_analysis import AttentionAnalyzer
        true_scores_list = []
        for pattern in true_patterns:
            scores = analyzer.compute_subject_attention_scores(pattern)
            true_scores_list.append(scores)
        true_scores_tensor = torch.stack(true_scores_list, dim=0)

        false_scores_list = []
        for pattern in false_patterns:
            scores = analyzer.compute_subject_attention_scores(pattern)
            false_scores_list.append(scores)
        false_scores_tensor = torch.stack(false_scores_list, dim=0)

        fig_dist = plot_head_scores_distribution(
            true_scores_tensor,
            false_scores_tensor,
            layer,
            head,
        )

        if args.save_html:
            fig_dist.write_html(output_dir / "score_distribution.html")
            logger.info(f"  Saved: {output_dir / 'score_distribution.html'}")
        else:
            fig_dist.show()

    # 6.6 Aggregated attention flow
    logger.info("Creating aggregated attention flow...")

    fig_agg = plot_aggregated_attention_flow(
        true_patterns,
        aggregation='mean',
        title=f"Mean Attention Flow (True Facts) - {args.model}",
    )

    if args.save_html:
        fig_agg.write_html(output_dir / "aggregated_attention.html")
        logger.info(f"  Saved: {output_dir / 'aggregated_attention.html'}")
    else:
        fig_agg.show()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("Analysis Complete!")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"True facts: {len(true_patterns)}")
    logger.info(f"False facts: {len(false_patterns)}")
    logger.info(f"Significant heads: {n_significant} / {model.cfg.n_layers * model.cfg.n_heads}")
    logger.info(f"  ({n_significant / (model.cfg.n_layers * model.cfg.n_heads) * 100:.1f}%)")

    if args.save_html:
        logger.info(f"\nVisualizations saved to: {output_dir}")
        logger.info("  - recall_heads_overview.html")
        logger.info("  - top_heads_comparison.html")
        logger.info("  - top_head_attention.html")
        logger.info("  - attention_comparison.html")
        logger.info("  - score_distribution.html")
        logger.info("  - aggregated_attention.html")

    logger.info("\nKey Findings:")
    if significant_heads:
        top_3 = significant_heads[:3]
        logger.info("  Top 3 factual recall heads:")
        for i, head in enumerate(top_3):
            logger.info(
                f"    {i+1}. L{head.layer}H{head.head} "
                f"(effect={head.score:.4f}, p={head.p_value:.2e})"
            )


if __name__ == "__main__":
    main()
