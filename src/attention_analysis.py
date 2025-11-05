"""Attention analysis for identifying factual recall heads.

This module provides tools for analyzing attention patterns in transformer models,
with a focus on identifying heads that are important for factual recall.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass
class AttentionPattern:
    """Stores attention patterns for a single prompt.

    Attributes:
        prompt: The input prompt
        tokens: List of token strings
        attention: Attention patterns [n_layers, n_heads, seq_len, seq_len]
        subject_positions: Token positions of the subject entity
        object_positions: Token positions of the object entity (if applicable)
        prediction_position: Position of the prediction token (usually last)
    """
    prompt: str
    tokens: List[str]
    attention: torch.Tensor
    subject_positions: Optional[List[int]] = None
    object_positions: Optional[List[int]] = None
    prediction_position: Optional[int] = None


@dataclass
class HeadScore:
    """Score for a single attention head.

    Attributes:
        layer: Layer index
        head: Head index
        score: Attention score from prediction to subject
        mean_score: Mean score across all prompts
        std_score: Standard deviation across prompts
        p_value: Statistical significance (if computed)
    """
    layer: int
    head: int
    score: float
    mean_score: Optional[float] = None
    std_score: Optional[float] = None
    p_value: Optional[float] = None


class AttentionAnalyzer:
    """Analyzes attention patterns for factual recall.

    Example:
        >>> model = HookedTransformer.from_pretrained('gpt2-medium')
        >>> analyzer = AttentionAnalyzer(model)
        >>> patterns = analyzer.extract_attention_patterns(["Paris is the capital of France"])
        >>> scores = analyzer.compute_subject_attention_scores(patterns[0])
    """

    def __init__(self, model: HookedTransformer):
        """Initialize the attention analyzer.

        Args:
            model: HookedTransformer model to analyze
        """
        self.model = model
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads

        logger.info(
            f"Initialized AttentionAnalyzer for {model.cfg.model_name} "
            f"({self.n_layers} layers, {self.n_heads} heads)"
        )

    def extract_attention_patterns(
        self,
        prompts: Union[str, List[str]],
        identify_subject: bool = True,
        subject_markers: Optional[List[str]] = None,
    ) -> List[AttentionPattern]:
        """Extract attention patterns for prompts.

        Args:
            prompts: Single prompt or list of prompts
            identify_subject: If True, attempt to identify subject positions
            subject_markers: Optional list of subject tokens to look for

        Returns:
            List of AttentionPattern objects
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        patterns = []

        for prompt in tqdm(prompts, desc="Extracting attention"):
            # Run model and get attention patterns
            tokens = self.model.to_str_tokens(prompt)
            _, cache = self.model.run_with_cache(prompt)

            # Stack attention patterns: [n_layers, n_heads, seq_len, seq_len]
            attention = torch.stack(
                [cache['pattern', layer] for layer in range(self.n_layers)],
                dim=0
            )[0]  # Remove batch dimension

            # Identify subject positions if requested
            subject_pos = None
            if identify_subject and subject_markers is not None:
                subject_pos = self._find_token_positions(tokens, subject_markers)

            pattern = AttentionPattern(
                prompt=prompt,
                tokens=tokens,
                attention=attention,
                subject_positions=subject_pos,
                prediction_position=len(tokens) - 1,  # Last token
            )

            patterns.append(pattern)

        return patterns

    def _find_token_positions(
        self,
        tokens: List[str],
        target_tokens: List[str],
    ) -> List[int]:
        """Find positions of target tokens in token list.

        Args:
            tokens: List of token strings
            target_tokens: Tokens to find

        Returns:
            List of positions where target tokens appear
        """
        positions = []

        # Normalize tokens for comparison
        tokens_lower = [t.lower().strip() for t in tokens]
        targets_lower = [t.lower().strip() for t in target_tokens]

        for i, token in enumerate(tokens_lower):
            if any(target in token or token in target for target in targets_lower):
                positions.append(i)

        return positions

    def compute_subject_attention_scores(
        self,
        pattern: AttentionPattern,
        aggregation: str = 'mean',
    ) -> torch.Tensor:
        """Compute attention scores from prediction token to subject.

        Args:
            pattern: AttentionPattern object
            aggregation: How to aggregate across subject positions ('mean', 'max', 'sum')

        Returns:
            Tensor of scores [n_layers, n_heads]
        """
        if pattern.subject_positions is None or len(pattern.subject_positions) == 0:
            logger.warning("No subject positions found, returning zeros")
            return torch.zeros(self.n_layers, self.n_heads)

        pred_pos = pattern.prediction_position
        subj_pos = pattern.subject_positions

        # Extract attention from prediction to subject positions
        # attention: [n_layers, n_heads, seq_len, seq_len]
        # We want: [n_layers, n_heads, len(subj_pos)]
        attn_to_subject = pattern.attention[:, :, pred_pos, subj_pos]

        # Aggregate across subject positions
        if aggregation == 'mean':
            scores = attn_to_subject.mean(dim=-1)
        elif aggregation == 'max':
            scores = attn_to_subject.max(dim=-1).values
        elif aggregation == 'sum':
            scores = attn_to_subject.sum(dim=-1)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return scores  # [n_layers, n_heads]

    def identify_factual_recall_heads(
        self,
        true_patterns: List[AttentionPattern],
        false_patterns: List[AttentionPattern],
        threshold: float = 0.05,
        min_effect_size: float = 0.1,
    ) -> Dict[str, Union[List[HeadScore], torch.Tensor]]:
        """Identify heads that are significantly more active for true facts.

        Args:
            true_patterns: Attention patterns for true facts
            false_patterns: Attention patterns for false facts
            threshold: p-value threshold for significance
            min_effect_size: Minimum effect size (difference in means)

        Returns:
            Dictionary containing:
                - 'significant_heads': List of HeadScore objects for significant heads
                - 'true_scores': Mean scores for true facts [n_layers, n_heads]
                - 'false_scores': Mean scores for false facts [n_layers, n_heads]
                - 'effect_sizes': Difference in means [n_layers, n_heads]
                - 'p_values': P-values from t-tests [n_layers, n_heads]
        """
        logger.info(f"Analyzing {len(true_patterns)} true and {len(false_patterns)} false facts")

        # Compute scores for all patterns
        true_scores_list = []
        false_scores_list = []

        for pattern in true_patterns:
            scores = self.compute_subject_attention_scores(pattern)
            true_scores_list.append(scores)

        for pattern in false_patterns:
            scores = self.compute_subject_attention_scores(pattern)
            false_scores_list.append(scores)

        # Stack: [n_samples, n_layers, n_heads]
        true_scores = torch.stack(true_scores_list, dim=0)
        false_scores = torch.stack(false_scores_list, dim=0)

        # Compute statistics for each head
        true_mean = true_scores.mean(dim=0)  # [n_layers, n_heads]
        false_mean = false_scores.mean(dim=0)

        effect_sizes = true_mean - false_mean

        # Perform t-tests for each head
        p_values = torch.zeros(self.n_layers, self.n_heads)

        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                true_vals = true_scores[:, layer, head].numpy()
                false_vals = false_scores[:, layer, head].numpy()

                # Two-sample t-test
                t_stat, p_val = stats.ttest_ind(true_vals, false_vals)
                p_values[layer, head] = p_val

        # Identify significant heads
        significant_heads = []

        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                p_val = p_values[layer, head].item()
                effect = effect_sizes[layer, head].item()

                # Check if significant and effect size is large enough
                if p_val < threshold and effect > min_effect_size:
                    head_score = HeadScore(
                        layer=layer,
                        head=head,
                        score=effect,
                        mean_score=true_mean[layer, head].item(),
                        std_score=true_scores[:, layer, head].std().item(),
                        p_value=p_val,
                    )
                    significant_heads.append(head_score)

        # Sort by effect size
        significant_heads.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Found {len(significant_heads)} significant heads")

        return {
            'significant_heads': significant_heads,
            'true_scores': true_mean,
            'false_scores': false_mean,
            'effect_sizes': effect_sizes,
            'p_values': p_values,
        }

    def compute_attention_statistics(
        self,
        patterns: List[AttentionPattern],
    ) -> Dict[str, torch.Tensor]:
        """Compute aggregate statistics across attention patterns.

        Args:
            patterns: List of AttentionPattern objects

        Returns:
            Dictionary of statistics:
                - 'mean_attention': Mean attention [n_layers, n_heads, seq_len, seq_len]
                - 'std_attention': Std attention
                - 'subject_scores_mean': Mean subject attention scores [n_layers, n_heads]
                - 'subject_scores_std': Std subject attention scores
        """
        # Compute subject scores for each pattern
        scores_list = []
        attention_list = []

        for pattern in patterns:
            scores = self.compute_subject_attention_scores(pattern)
            scores_list.append(scores)
            attention_list.append(pattern.attention)

        # Stack and compute statistics
        all_scores = torch.stack(scores_list, dim=0)  # [n_patterns, n_layers, n_heads]

        # For attention patterns, we need to handle variable sequence lengths
        # So we'll compute statistics on subject scores instead

        return {
            'subject_scores_mean': all_scores.mean(dim=0),
            'subject_scores_std': all_scores.std(dim=0),
            'subject_scores_all': all_scores,
        }

    def get_top_heads(
        self,
        patterns: List[AttentionPattern],
        top_k: int = 10,
    ) -> List[HeadScore]:
        """Get top-k heads by average subject attention.

        Args:
            patterns: List of AttentionPattern objects
            top_k: Number of top heads to return

        Returns:
            List of HeadScore objects, sorted by score
        """
        stats = self.compute_attention_statistics(patterns)
        mean_scores = stats['subject_scores_mean']
        std_scores = stats['subject_scores_std']

        # Flatten and get top-k
        scores_flat = mean_scores.flatten()
        top_indices = scores_flat.topk(top_k).indices

        top_heads = []
        for idx in top_indices:
            layer = idx // self.n_heads
            head = idx % self.n_heads

            head_score = HeadScore(
                layer=layer.item(),
                head=head.item(),
                score=mean_scores[layer, head].item(),
                mean_score=mean_scores[layer, head].item(),
                std_score=std_scores[layer, head].item(),
            )
            top_heads.append(head_score)

        return top_heads

    def analyze_attention_flow(
        self,
        pattern: AttentionPattern,
        source_positions: List[int],
        target_position: int,
    ) -> torch.Tensor:
        """Analyze attention flow from source to target positions.

        Args:
            pattern: AttentionPattern object
            source_positions: Source token positions
            target_position: Target token position

        Returns:
            Attention scores [n_layers, n_heads, len(source_positions)]
        """
        # attention: [n_layers, n_heads, seq_len, seq_len]
        attention_flow = pattern.attention[:, :, target_position, source_positions]
        return attention_flow


def compare_attention_distributions(
    true_scores: torch.Tensor,
    false_scores: torch.Tensor,
    test: str = 'ttest',
) -> Dict[str, torch.Tensor]:
    """Compare attention score distributions using statistical tests.

    Args:
        true_scores: Scores for true facts [n_samples, n_layers, n_heads]
        false_scores: Scores for false facts [n_samples, n_layers, n_heads]
        test: Statistical test to use ('ttest', 'mannwhitneyu')

    Returns:
        Dictionary with 'statistics' and 'p_values' tensors
    """
    n_layers, n_heads = true_scores.shape[1], true_scores.shape[2]

    statistics = torch.zeros(n_layers, n_heads)
    p_values = torch.zeros(n_layers, n_heads)

    for layer in range(n_layers):
        for head in range(n_heads):
            true_vals = true_scores[:, layer, head].numpy()
            false_vals = false_scores[:, layer, head].numpy()

            if test == 'ttest':
                stat, p_val = stats.ttest_ind(true_vals, false_vals)
            elif test == 'mannwhitneyu':
                stat, p_val = stats.mannwhitneyu(true_vals, false_vals, alternative='two-sided')
            else:
                raise ValueError(f"Unknown test: {test}")

            statistics[layer, head] = stat
            p_values[layer, head] = p_val

    return {
        'statistics': statistics,
        'p_values': p_values,
    }


def compute_bonferroni_correction(
    p_values: torch.Tensor,
    alpha: float = 0.05,
) -> Tuple[torch.Tensor, float]:
    """Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: P-values [n_layers, n_heads]
        alpha: Significance level

    Returns:
        Tuple of (significant_mask, corrected_threshold)
    """
    n_comparisons = p_values.numel()
    corrected_alpha = alpha / n_comparisons

    significant_mask = p_values < corrected_alpha

    return significant_mask, corrected_alpha


def compute_fdr_correction(
    p_values: torch.Tensor,
    alpha: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: P-values [n_layers, n_heads]
        alpha: FDR level

    Returns:
        Tuple of (significant_mask, adjusted_p_values)
    """
    # Flatten p-values
    p_flat = p_values.flatten()
    n = len(p_flat)

    # Sort p-values
    sorted_p, sorted_indices = torch.sort(p_flat)

    # Compute critical values
    ranks = torch.arange(1, n + 1, dtype=torch.float32)
    critical_values = (ranks / n) * alpha

    # Find largest i where p[i] <= critical_value[i]
    significant = sorted_p <= critical_values

    if significant.any():
        max_idx = significant.nonzero(as_tuple=True)[0][-1]
        threshold = sorted_p[max_idx]
    else:
        threshold = 0.0

    # Create mask
    significant_mask = p_values <= threshold

    # Compute adjusted p-values (Benjamini-Hochberg)
    adjusted_p = torch.zeros_like(p_flat)
    for i in range(n):
        original_idx = sorted_indices[i]
        adjusted_p[original_idx] = min(1.0, sorted_p[i] * n / (i + 1))

    adjusted_p = adjusted_p.reshape(p_values.shape)

    return significant_mask, adjusted_p
