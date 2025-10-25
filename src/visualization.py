"""Visualization utilities for mechanistic interpretability analysis."""

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from transformer_lens import HookedTransformer
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import circuitsvis for interactive attention viz
try:
    import circuitsvis as cv
    HAS_CIRCUITSVIS = True
except ImportError:
    HAS_CIRCUITSVIS = False
    logger.warning("circuitsvis not installed, some visualizations unavailable")


def plot_attention_pattern(
    tokens: List[str],
    attention: np.ndarray,
    title: str = "Attention Pattern",
    layer: Optional[int] = None,
    head: Optional[int] = None,
) -> go.Figure:
    """
    Plot attention pattern heatmap.

    Args:
        tokens: List of token strings
        attention: Attention weights [seq_len, seq_len]
        title: Plot title
        layer: Optional layer number for title
        head: Optional head number for title

    Returns:
        Plotly figure
    """
    if layer is not None and head is not None:
        title = f"{title} - Layer {layer} Head {head}"

    fig = go.Figure(data=go.Heatmap(
        z=attention,
        x=tokens,
        y=tokens,
        colorscale="Blues",
        zmin=0,
        zmax=1,
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Key",
        yaxis_title="Query",
        width=600,
        height=600,
    )

    return fig


def plot_attention_heads(
    model: HookedTransformer,
    prompt: str,
    layer: int,
    max_heads: Optional[int] = None,
) -> go.Figure:
    """
    Plot attention patterns for all heads in a layer.

    Args:
        model: HookedTransformer model
        prompt: Input prompt
        layer: Layer index
        max_heads: Optional limit on number of heads to plot

    Returns:
        Plotly figure with subplots
    """
    tokens = model.to_str_tokens(prompt)
    _, cache = model.run_with_cache(prompt)
    attention = cache["pattern", layer][0].cpu().numpy()  # [n_heads, seq_len, seq_len]

    n_heads = model.cfg.n_heads if max_heads is None else min(max_heads, model.cfg.n_heads)
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Head {h}" for h in range(n_heads)],
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    )

    for head in range(n_heads):
        row = head // cols + 1
        col = head % cols + 1

        fig.add_trace(
            go.Heatmap(
                z=attention[head],
                x=tokens,
                y=tokens,
                colorscale="Blues",
                showscale=(head == 0),
                zmin=0,
                zmax=1,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"Attention Patterns - Layer {layer}",
        height=300 * rows,
        width=300 * cols,
    )

    return fig


def plot_head_effects(
    effects: torch.Tensor,
    title: str = "Head Effects",
    colorscale: str = "RdBu",
) -> go.Figure:
    """
    Plot heatmap of head effects (ablation, patching, etc.).

    Args:
        effects: Tensor of shape [n_layers, n_heads]
        title: Plot title
        colorscale: Plotly colorscale

    Returns:
        Plotly figure
    """
    n_layers, n_heads = effects.shape

    fig = go.Figure(data=go.Heatmap(
        z=effects.numpy(),
        x=[f"H{h}" for h in range(n_heads)],
        y=[f"L{l}" for l in range(n_layers)],
        colorscale=colorscale,
        zmid=0,
        colorbar=dict(title="Effect"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Head",
        yaxis_title="Layer",
        width=max(600, n_heads * 40),
        height=max(400, n_layers * 40),
    )

    return fig


def plot_component_effects(
    effects: Dict[str, float],
    top_k: Optional[int] = 20,
    title: str = "Component Effects",
) -> go.Figure:
    """
    Plot bar chart of component effects.

    Args:
        effects: Dictionary mapping component name -> effect
        top_k: Optional limit on number of components to show
        title: Plot title

    Returns:
        Plotly figure
    """
    # Sort by absolute effect
    sorted_effects = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)

    if top_k is not None:
        sorted_effects = sorted_effects[:top_k]

    components, values = zip(*sorted_effects)

    fig = go.Figure(data=go.Bar(
        x=list(components),
        y=list(values),
        marker_color=["red" if v < 0 else "blue" for v in values],
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Component",
        yaxis_title="Effect",
        width=max(800, len(components) * 30),
        height=500,
    )

    return fig


def plot_activation_patching_results(
    results: Dict[Tuple[str, int], float],
    n_layers: int,
    components: List[str],
) -> go.Figure:
    """
    Plot activation patching results as heatmap.

    Args:
        results: Dictionary from (component, layer) -> effect
        n_layers: Number of layers
        components: List of component names

    Returns:
        Plotly figure
    """
    # Organize into matrix
    matrix = np.zeros((len(components), n_layers))

    for (comp, layer), effect in results.items():
        comp_idx = components.index(comp)
        matrix[comp_idx, layer] = effect

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"L{l}" for l in range(n_layers)],
        y=components,
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="Patching Effect"),
    ))

    fig.update_layout(
        title="Activation Patching Results",
        xaxis_title="Layer",
        yaxis_title="Component",
        width=max(800, n_layers * 40),
        height=max(400, len(components) * 50),
    )

    return fig


def plot_logit_lens(
    model: HookedTransformer,
    prompt: str,
    top_k: int = 10,
) -> go.Figure:
    """
    Plot logit lens: decode residual stream at each layer.

    Args:
        model: HookedTransformer model
        prompt: Input prompt
        top_k: Number of top predictions to show

    Returns:
        Plotly figure
    """
    _, cache = model.run_with_cache(prompt)

    # Get residual stream at each layer
    n_layers = model.cfg.n_layers
    top_predictions = []

    for layer in range(n_layers + 1):  # +1 for final output
        if layer == n_layers:
            resid = cache["resid_post", layer - 1]
        else:
            resid = cache["resid_post", layer]

        # Decode to vocabulary
        logits = model.unembed(model.ln_final(resid))  # [batch, seq, vocab]
        final_logits = logits[0, -1, :]  # Final position

        # Get top-k predictions
        top_vals, top_indices = final_logits.topk(top_k)
        top_tokens = [model.tokenizer.decode(idx) for idx in top_indices]

        top_predictions.append((layer, top_tokens, top_vals.cpu().numpy()))

    # Create visualization
    fig = go.Figure()

    for layer, tokens, values in top_predictions:
        fig.add_trace(go.Bar(
            name=f"Layer {layer}",
            x=tokens,
            y=values,
            visible=(layer == n_layers),  # Show final layer by default
        ))

    # Add slider
    steps = []
    for i, (layer, _, _) in enumerate(top_predictions):
        step = dict(
            method="update",
            args=[{"visible": [j == i for j in range(len(top_predictions))]}],
            label=f"L{layer}",
        )
        steps.append(step)

    sliders = [dict(
        active=len(top_predictions) - 1,
        steps=steps,
        currentvalue={"prefix": "Layer: "},
    )]

    fig.update_layout(
        title=f"Logit Lens - '{prompt}'",
        xaxis_title="Token",
        yaxis_title="Logit",
        sliders=sliders,
        width=800,
        height=500,
    )

    return fig


def plot_neuron_activations(
    activations: torch.Tensor,
    tokens: List[str],
    top_k: int = 20,
    title: str = "Top Neuron Activations",
) -> go.Figure:
    """
    Plot top activated neurons across a sequence.

    Args:
        activations: Neuron activations [seq_len, n_neurons]
        tokens: Token strings
        top_k: Number of top neurons to show
        title: Plot title

    Returns:
        Plotly figure
    """
    # Find top-k most activated neurons (max across sequence)
    max_acts, _ = activations.max(dim=0)
    top_neurons = max_acts.topk(top_k).indices

    # Get activations for these neurons
    top_acts = activations[:, top_neurons].T.cpu().numpy()  # [top_k, seq_len]

    fig = go.Figure(data=go.Heatmap(
        z=top_acts,
        x=tokens,
        y=[f"N{n.item()}" for n in top_neurons],
        colorscale="Reds",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Token",
        yaxis_title="Neuron",
        width=max(800, len(tokens) * 50),
        height=max(400, top_k * 20),
    )

    return fig


def visualize_attention_interactive(
    model: HookedTransformer,
    prompt: str,
    layer: int,
    head: int,
):
    """
    Create interactive attention visualization using CircuitsVis.

    Args:
        model: HookedTransformer model
        prompt: Input prompt
        layer: Layer index
        head: Head index

    Returns:
        CircuitsVis HTML visualization (or None if unavailable)
    """
    if not HAS_CIRCUITSVIS:
        logger.warning("circuitsvis not installed, falling back to static plot")
        return None

    tokens = model.to_str_tokens(prompt)
    _, cache = model.run_with_cache(prompt)
    attention = cache["pattern", layer][0, head].cpu().numpy()  # [seq_len, seq_len]

    return cv.attention.attention_patterns(
        tokens=tokens,
        attention=attention,
    )


def plot_circuit_graph(
    important_heads: set,
    n_layers: int,
    n_heads: int,
    title: str = "Discovered Circuit",
) -> go.Figure:
    """
    Plot circuit as a graph of important components.

    Args:
        important_heads: Set of (layer, head) tuples
        n_layers: Number of layers
        n_heads: Number of heads per layer
        title: Plot title

    Returns:
        Plotly figure
    """
    # Create scatter plot with head positions
    layers = []
    heads = []
    colors = []

    for layer in range(n_layers):
        for head in range(n_heads):
            layers.append(layer)
            heads.append(head)
            colors.append("red" if (layer, head) in important_heads else "lightgray")

    fig = go.Figure(data=go.Scatter(
        x=layers,
        y=heads,
        mode="markers",
        marker=dict(
            color=colors,
            size=15,
            line=dict(width=1, color="black"),
        ),
        text=[f"L{l}H{h}" for l, h in zip(layers, heads)],
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Head",
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        yaxis=dict(tickmode="linear", tick0=0, dtick=1),
        width=max(800, n_layers * 100),
        height=max(600, n_heads * 50),
    )

    return fig
