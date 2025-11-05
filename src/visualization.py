"""Visualization utilities for mechanistic interpretability analysis."""

import logging

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)

# Try to import UMAP
try:
    from umap import UMAP

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logger.warning("umap-learn not installed, UMAP visualizations unavailable")

# Try to import circuitsvis for interactive attention viz
try:
    import circuitsvis as cv

    HAS_CIRCUITSVIS = True
except ImportError:
    HAS_CIRCUITSVIS = False
    logger.warning("circuitsvis not installed, some visualizations unavailable")


def plot_attention_pattern(
    tokens: list[str],
    attention: np.ndarray,
    title: str = "Attention Pattern",
    layer: int | None = None,
    head: int | None = None,
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

    fig = go.Figure(
        data=go.Heatmap(
            z=attention,
            x=tokens,
            y=tokens,
            colorscale="Blues",
            zmin=0,
            zmax=1,
        )
    )

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
    max_heads: int | None = None,
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

    fig = go.Figure(
        data=go.Heatmap(
            z=effects.numpy(),
            x=[f"H{h}" for h in range(n_heads)],
            y=[f"L{l}" for l in range(n_layers)],
            colorscale=colorscale,
            zmid=0,
            colorbar={"title": "Effect"},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Head",
        yaxis_title="Layer",
        width=max(600, n_heads * 40),
        height=max(400, n_layers * 40),
    )

    return fig


def plot_component_effects(
    effects: dict[str, float],
    top_k: int | None = 20,
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

    components, values = zip(*sorted_effects, strict=False)

    fig = go.Figure(
        data=go.Bar(
            x=list(components),
            y=list(values),
            marker_color=["red" if v < 0 else "blue" for v in values],
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Component",
        yaxis_title="Effect",
        width=max(800, len(components) * 30),
        height=500,
    )

    return fig


def plot_activation_patching_results(
    results: dict[tuple[str, int], float],
    n_layers: int,
    components: list[str],
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

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"L{l}" for l in range(n_layers)],
            y=components,
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "Patching Effect"},
        )
    )

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
        fig.add_trace(
            go.Bar(
                name=f"Layer {layer}",
                x=tokens,
                y=values,
                visible=(layer == n_layers),  # Show final layer by default
            )
        )

    # Add slider
    steps = []
    for i, (layer, _, _) in enumerate(top_predictions):
        step = {
            "method": "update",
            "args": [{"visible": [j == i for j in range(len(top_predictions))]}],
            "label": f"L{layer}",
        }
        steps.append(step)

    sliders = [
        {
            "active": len(top_predictions) - 1,
            "steps": steps,
            "currentvalue": {"prefix": "Layer: "},
        }
    ]

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
    tokens: list[str],
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

    fig = go.Figure(
        data=go.Heatmap(
            z=top_acts,
            x=tokens,
            y=[f"N{n.item()}" for n in top_neurons],
            colorscale="Reds",
        )
    )

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

    fig = go.Figure(
        data=go.Scatter(
            x=layers,
            y=heads,
            mode="markers",
            marker={
                "color": colors,
                "size": 15,
                "line": {"width": 1, "color": "black"},
            },
            text=[f"L{l}H{h}" for l, h in zip(layers, heads, strict=False)],
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Head",
        xaxis={"tickmode": "linear", "tick0": 0, "dtick": 1},
        yaxis={"tickmode": "linear", "tick0": 0, "dtick": 1},
        width=max(800, n_layers * 100),
        height=max(600, n_heads * 50),
    )

    return fig


def plot_activation_magnitude_heatmap(
    activations: torch.Tensor,
    labels: list[str] | None = None,
    title: str = "Activation Magnitudes Across Layers",
    metric: str = "l2_norm",
) -> go.Figure:
    """Plot heatmap of activation magnitudes across layers and prompts.

    Args:
        activations: Tensor of shape [n_prompts, n_layers, seq_len, d_model] or
                    [n_prompts, n_layers, d_model]
        labels: Optional labels for each prompt
        title: Plot title
        metric: How to aggregate activations. Options:
               'l2_norm': L2 norm of activations
               'mean': Mean absolute value
               'max': Maximum absolute value
               'std': Standard deviation

    Returns:
        Plotly heatmap figure
    """
    # Convert to numpy
    acts = activations.detach().cpu().numpy()

    # Handle different input shapes
    if len(acts.shape) == 4:  # [n_prompts, n_layers, seq_len, d_model]
        # Aggregate over sequence dimension
        if metric == "l2_norm":
            # Compute L2 norm over d_model, then mean over seq_len
            values = np.linalg.norm(acts, axis=-1).mean(axis=-1)
        elif metric == "mean":
            values = np.abs(acts).mean(axis=(-2, -1))
        elif metric == "max":
            values = np.abs(acts).max(axis=-1).mean(axis=-1)
        elif metric == "std":
            values = acts.std(axis=(-2, -1))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    elif len(acts.shape) == 3:  # [n_prompts, n_layers, d_model]
        if metric == "l2_norm":
            values = np.linalg.norm(acts, axis=-1)
        elif metric == "mean":
            values = np.abs(acts).mean(axis=-1)
        elif metric == "max":
            values = np.abs(acts).max(axis=-1)
        elif metric == "std":
            values = acts.std(axis=-1)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        raise ValueError(f"Unexpected activation shape: {acts.shape}")

    n_prompts, n_layers = values.shape

    # Create labels if not provided
    if labels is None:
        labels = [f"Prompt {i}" for i in range(n_prompts)]

    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=[f"L{i}" for i in range(n_layers)],
            y=labels,
            colorscale="Viridis",
            colorbar={"title": metric.replace("_", " ").title()},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Prompt",
        width=max(800, n_layers * 50),
        height=max(400, n_prompts * 30),
    )

    return fig


def plot_activation_comparison(
    true_activations: torch.Tensor,
    false_activations: torch.Tensor,
    layer_idx: int | None = None,
    title: str = "True vs False Fact Activations",
) -> go.Figure:
    """Compare activation magnitudes for true vs false facts.

    Args:
        true_activations: Activations for true facts [n_true, n_layers, d_model]
        false_activations: Activations for false facts [n_false, n_layers, d_model]
        layer_idx: Optional specific layer to visualize. If None, shows all layers.
        title: Plot title

    Returns:
        Plotly figure
    """
    true_acts = true_activations.detach().cpu()
    false_acts = false_activations.detach().cpu()

    # Compute L2 norms
    true_norms = torch.norm(true_acts, dim=-1)  # [n_true, n_layers]
    false_norms = torch.norm(false_acts, dim=-1)  # [n_false, n_layers]

    if layer_idx is not None:
        # Plot distribution for specific layer
        fig = go.Figure()

        fig.add_trace(
            go.Box(
                y=true_norms[:, layer_idx].numpy(),
                name="True Facts",
                marker_color="green",
            )
        )

        fig.add_trace(
            go.Box(
                y=false_norms[:, layer_idx].numpy(),
                name="False Facts",
                marker_color="red",
            )
        )

        fig.update_layout(
            title=f"{title} - Layer {layer_idx}",
            yaxis_title="L2 Norm",
            showlegend=True,
            width=600,
            height=500,
        )

    else:
        # Plot means across layers with error bars
        true_mean = true_norms.mean(dim=0).numpy()
        true_std = true_norms.std(dim=0).numpy()
        false_mean = false_norms.mean(dim=0).numpy()
        false_std = false_norms.std(dim=0).numpy()

        n_layers = true_mean.shape[0]
        layers = list(range(n_layers))

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=layers,
                y=true_mean,
                error_y={"type": "data", "array": true_std},
                mode="lines+markers",
                name="True Facts",
                line={"color": "green", "width": 2},
                marker={"size": 8},
            )
        )

        fig.add_trace(
            go.Scatter(
                x=layers,
                y=false_mean,
                error_y={"type": "data", "array": false_std},
                mode="lines+markers",
                name="False Facts",
                line={"color": "red", "width": 2},
                marker={"size": 8},
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Layer",
            yaxis_title="Mean L2 Norm",
            showlegend=True,
            width=900,
            height=500,
        )

    return fig


def plot_pca_activations(
    activations: torch.Tensor,
    labels: list[str | int] | None = None,
    colors: list[str | int] | None = None,
    n_components: int = 2,
    layer_idx: int | None = None,
    title: str = "PCA of Activations",
) -> go.Figure:
    """Visualize activations in PCA space.

    Args:
        activations: Tensor of shape [n_samples, n_layers, d_model] or
                    [n_samples, d_model]
        labels: Optional labels for each sample (for hover text)
        colors: Optional color values for each sample (can be categorical or continuous)
        n_components: Number of PCA components (2 or 3)
        layer_idx: If activations have layer dimension, which layer to use
        title: Plot title

    Returns:
        Plotly scatter plot (2D or 3D)
    """
    acts = activations.detach().cpu().numpy()

    # Handle layer dimension
    if len(acts.shape) == 3:
        if layer_idx is None:
            # Average across layers
            acts = acts.mean(axis=1)
        else:
            acts = acts[:, layer_idx, :]

    n_samples = acts.shape[0]

    # Apply PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(acts)

    # Prepare labels and colors
    if labels is None:
        labels = [f"Sample {i}" for i in range(n_samples)]

    if colors is None:
        colors = ["blue"] * n_samples

    # Create plot
    if n_components == 2:
        fig = go.Figure(
            data=go.Scatter(
                x=transformed[:, 0],
                y=transformed[:, 1],
                mode="markers",
                marker={
                    "size": 10,
                    "color": colors,
                    "colorscale": "Viridis" if isinstance(colors[0], (int, float)) else None,
                    "showscale": isinstance(colors[0], (int, float)),
                    "line": {"width": 1, "color": "white"},
                },
                text=labels,
                hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"{title}<br>Explained variance: "
            f"PC1={pca.explained_variance_ratio_[0]:.2%}, "
            f"PC2={pca.explained_variance_ratio_[1]:.2%}",
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
            width=800,
            height=700,
        )

    else:  # 3D
        fig = go.Figure(
            data=go.Scatter3d(
                x=transformed[:, 0],
                y=transformed[:, 1],
                z=transformed[:, 2],
                mode="markers",
                marker={
                    "size": 6,
                    "color": colors,
                    "colorscale": "Viridis" if isinstance(colors[0], (int, float)) else None,
                    "showscale": isinstance(colors[0], (int, float)),
                    "line": {"width": 0.5, "color": "white"},
                },
                text=labels,
                hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>"
                "PC3: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"{title}<br>Explained variance: "
            f"{pca.explained_variance_ratio_.sum():.2%} total",
            scene={
                "xaxis_title": f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                "yaxis_title": f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                "zaxis_title": f"PC3 ({pca.explained_variance_ratio_[2]:.1%})",
            },
            width=900,
            height=700,
        )

    return fig


def plot_umap_activations(
    activations: torch.Tensor,
    labels: list[str | int] | None = None,
    colors: list[str | int] | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    layer_idx: int | None = None,
    title: str = "UMAP of Activations",
) -> go.Figure:
    """Visualize activations in UMAP space.

    Args:
        activations: Tensor of shape [n_samples, n_layers, d_model] or
                    [n_samples, d_model]
        labels: Optional labels for each sample
        colors: Optional color values for each sample
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        layer_idx: If activations have layer dimension, which layer to use
        title: Plot title

    Returns:
        Plotly scatter plot

    Raises:
        ImportError: If umap-learn is not installed
    """
    if not HAS_UMAP:
        raise ImportError("umap-learn is not installed. Install it with: pip install umap-learn")

    acts = activations.detach().cpu().numpy()

    # Handle layer dimension
    if len(acts.shape) == 3:
        if layer_idx is None:
            acts = acts.mean(axis=1)
        else:
            acts = acts[:, layer_idx, :]

    n_samples = acts.shape[0]

    # Apply UMAP
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)
    transformed = reducer.fit_transform(acts)

    # Prepare labels and colors
    if labels is None:
        labels = [f"Sample {i}" for i in range(n_samples)]

    if colors is None:
        colors = ["blue"] * n_samples

    fig = go.Figure(
        data=go.Scatter(
            x=transformed[:, 0],
            y=transformed[:, 1],
            mode="markers",
            marker={
                "size": 10,
                "color": colors,
                "colorscale": "Viridis" if isinstance(colors[0], (int, float)) else None,
                "showscale": isinstance(colors[0], (int, float)),
                "line": {"width": 1, "color": "white"},
            },
            text=labels,
            hovertemplate="<b>%{text}</b><br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        width=800,
        height=700,
    )

    return fig


def plot_activation_space_comparison(
    true_activations: torch.Tensor,
    false_activations: torch.Tensor,
    method: str = "pca",
    layer_idx: int | None = None,
    title: str | None = None,
    **kwargs,
) -> go.Figure:
    """Compare activation spaces for true vs false facts using dimensionality reduction.

    Args:
        true_activations: Activations for true facts
        false_activations: Activations for false facts
        method: Dimensionality reduction method ('pca', 'tsne', or 'umap')
        layer_idx: Optional layer index to visualize
        title: Optional custom title
        **kwargs: Additional arguments for the reduction method

    Returns:
        Plotly figure
    """
    # Combine activations
    all_acts = torch.cat([true_activations, false_activations], dim=0)

    # Create labels and colors
    n_true = true_activations.shape[0]
    n_false = false_activations.shape[0]

    labels = [f"True {i}" for i in range(n_true)] + [f"False {i}" for i in range(n_false)]

    # Use categorical colors: 0 for true, 1 for false
    colors = [True] * n_true + [False] * n_false

    if title is None:
        title = f"{method.upper()}: True vs False Fact Activations"

    # Apply dimensionality reduction
    if method.lower() == "pca":
        fig = plot_pca_activations(
            all_acts, labels=labels, colors=colors, layer_idx=layer_idx, title=title, **kwargs
        )
    elif method.lower() == "umap":
        fig = plot_umap_activations(
            all_acts, labels=labels, colors=colors, layer_idx=layer_idx, title=title, **kwargs
        )
    elif method.lower() == "tsne":
        fig = plot_tsne_activations(
            all_acts, labels=labels, colors=colors, layer_idx=layer_idx, title=title, **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'umap'")

    # Update colors to be categorical
    fig.data[0].marker.color = [1 if c == "True" else 0 for c in colors]
    fig.data[0].marker.colorscale = [[0, "red"], [1, "green"]]
    fig.data[0].marker.showscale = True
    fig.data[0].marker.colorbar = {
        "title": "Label",
        "tickvals": [0.25, 0.75],
        "ticktext": ["False", "True"],
    }

    return fig


def plot_tsne_activations(
    activations: torch.Tensor,
    labels: list[str | int] | None = None,
    colors: list[str | int] | None = None,
    perplexity: int = 30,
    layer_idx: int | None = None,
    title: str = "t-SNE of Activations",
) -> go.Figure:
    """Visualize activations in t-SNE space.

    Args:
        activations: Tensor of shape [n_samples, n_layers, d_model] or
                    [n_samples, d_model]
        labels: Optional labels for each sample
        colors: Optional color values for each sample
        perplexity: t-SNE perplexity parameter
        layer_idx: If activations have layer dimension, which layer to use
        title: Plot title

    Returns:
        Plotly scatter plot
    """
    acts = activations.detach().cpu().numpy()

    # Handle layer dimension
    if len(acts.shape) == 3:
        if layer_idx is None:
            acts = acts.mean(axis=1)
        else:
            acts = acts[:, layer_idx, :]

    n_samples = acts.shape[0]

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=min(perplexity, n_samples - 1))
    transformed = tsne.fit_transform(acts)

    # Prepare labels and colors
    if labels is None:
        labels = [f"Sample {i}" for i in range(n_samples)]

    if colors is None:
        colors = ["blue"] * n_samples

    fig = go.Figure(
        data=go.Scatter(
            x=transformed[:, 0],
            y=transformed[:, 1],
            mode="markers",
            marker={
                "size": 10,
                "color": colors,
                "colorscale": "Viridis" if isinstance(colors[0], (int, float)) else None,
                "showscale": isinstance(colors[0], (int, float)),
                "line": {"width": 1, "color": "white"},
            },
            text=labels,
            hovertemplate="<b>%{text}</b><br>t-SNE1: %{x:.3f}<br>t-SNE2: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        width=800,
        height=700,
    )

    return fig


def plot_factual_recall_heads(
    results: dict,
    top_k: int = 20,
    show_all: bool = False,
    title: str = "Factual Recall Heads Analysis",
) -> go.Figure:
    """Plot results from factual recall head analysis.

    Args:
        results: Results dictionary from identify_factual_recall_heads()
        top_k: Number of top heads to highlight
        show_all: If True, show all heads; otherwise only significant ones
        title: Plot title

    Returns:
        Plotly figure with heatmap and significance markers
    """
    effect_sizes = results["effect_sizes"]
    p_values = results["p_values"]
    significant_heads = results["significant_heads"]

    n_layers, n_heads = effect_sizes.shape

    # Create figure with subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Effect Size (True - False)", "Statistical Significance (-log10 p-value)"),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5],
    )

    # Plot 1: Effect sizes heatmap
    fig.add_trace(
        go.Heatmap(
            z=effect_sizes.numpy(),
            x=[f"H{h}" for h in range(n_heads)],
            y=[f"L{l}" for l in range(n_layers)],
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "Effect Size", "y": 0.75, "len": 0.4},
            hovertemplate="Layer %{y}<br>Head %{x}<br>Effect: %{z:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add markers for significant heads
    if significant_heads:
        sig_layers = [h.layer for h in significant_heads[:top_k]]
        sig_heads = [h.head for h in significant_heads[:top_k]]

        fig.add_trace(
            go.Scatter(
                x=[f"H{h}" for h in sig_heads],
                y=[f"L{l}" for l in sig_layers],
                mode="markers",
                marker={
                    "symbol": "star",
                    "size": 12,
                    "color": "yellow",
                    "line": {"width": 1, "color": "black"},
                },
                name="Top significant",
                hovertext=[
                    f"L{h.layer}H{h.head}<br>p={h.p_value:.2e}" for h in significant_heads[:top_k]
                ],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    # Plot 2: P-values (-log10)
    log_p_values = -torch.log10(p_values + 1e-10).numpy()

    fig.add_trace(
        go.Heatmap(
            z=log_p_values,
            x=[f"H{h}" for h in range(n_heads)],
            y=[f"L{l}" for l in range(n_layers)],
            colorscale="Viridis",
            colorbar={"title": "-log10(p)", "y": 0.25, "len": 0.4},
            hovertemplate="Layer %{y}<br>Head %{x}<br>-log10(p): %{z:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Add significance threshold line (p=0.05 -> -log10(p) ≈ 1.3)
    # TODO
    # threshold_line = -np.log10(0.05)

    fig.update_layout(
        title=title,
        height=800,
        width=max(800, n_heads * 40),
        showlegend=True,
    )
    # fig.add_hline(threshold_line)

    return fig


def plot_attention_to_subject(
    pattern,
    layer: int,
    head: int,
    title: str | None = None,
) -> go.Figure:
    """Plot attention pattern for a specific head, highlighting subject attention.

    Args:
        pattern: AttentionPattern object from attention_analysis
        layer: Layer index
        head: Head index
        title: Optional title

    Returns:
        Plotly heatmap figure
    """
    tokens = pattern.tokens
    attention = pattern.attention[layer, head].cpu().numpy()

    if title is None:
        title = f"Attention Pattern - Layer {layer} Head {head}"

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=attention,
            x=tokens,
            y=tokens,
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar={"title": "Attention"},
        )
    )

    # Highlight subject positions if available
    if pattern.subject_positions:
        for pos in pattern.subject_positions:
            # Add vertical line for subject token
            fig.add_vline(x=pos, line_width=2, line_dash="dash", line_color="red", opacity=0.5)

    # Highlight prediction position
    if pattern.prediction_position is not None:
        fig.add_hline(
            y=pattern.prediction_position,
            line_width=2,
            line_dash="dash",
            line_color="green",
            opacity=0.5,
        )

    fig.update_layout(
        title=title,
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        width=max(600, len(tokens) * 30),
        height=max(600, len(tokens) * 30),
    )

    return fig


def plot_attention_comparison_interactive(
    true_pattern,
    false_pattern,
    layer: int,
    head: int,
) -> go.Figure:
    """Create side-by-side comparison of attention for true vs false facts.

    Args:
        true_pattern: AttentionPattern for true fact
        false_pattern: AttentionPattern for false fact
        layer: Layer index
        head: Head index

    Returns:
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("True Fact", "False Fact"),
        horizontal_spacing=0.1,
    )

    # True fact attention
    true_attn = true_pattern.attention[layer, head].cpu().numpy()
    fig.add_trace(
        go.Heatmap(
            z=true_attn,
            x=true_pattern.tokens,
            y=true_pattern.tokens,
            colorscale="Blues",
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar={"x": 0.45, "title": "Attention"},
        ),
        row=1,
        col=1,
    )

    # False fact attention
    false_attn = false_pattern.attention[layer, head].cpu().numpy()
    fig.add_trace(
        go.Heatmap(
            z=false_attn,
            x=false_pattern.tokens,
            y=false_pattern.tokens,
            colorscale="Blues",
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar={"x": 1.05, "title": "Attention"},
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"Attention Comparison - Layer {layer} Head {head}",
        height=500,
        width=1200,
    )

    return fig


def plot_head_scores_distribution(
    true_scores: torch.Tensor,
    false_scores: torch.Tensor,
    layer: int,
    head: int,
    title: str | None = None,
) -> go.Figure:
    """Plot distribution of attention scores for a specific head.

    Args:
        true_scores: Scores for true facts [n_samples, n_layers, n_heads]
        false_scores: Scores for false facts [n_samples, n_layers, n_heads]
        layer: Layer index
        head: Head index
        title: Optional title

    Returns:
        Plotly figure with overlaid distributions
    """
    true_vals = true_scores[:, layer, head].numpy()
    false_vals = false_scores[:, layer, head].numpy()

    if title is None:
        title = f"Score Distribution - Layer {layer} Head {head}"

    fig = go.Figure()

    # Add histograms
    fig.add_trace(
        go.Histogram(
            x=true_vals,
            name="True Facts",
            opacity=0.7,
            marker_color="green",
            nbinsx=20,
        )
    )

    fig.add_trace(
        go.Histogram(
            x=false_vals,
            name="False Facts",
            opacity=0.7,
            marker_color="red",
            nbinsx=20,
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Attention Score to Subject",
        yaxis_title="Count",
        barmode="overlay",
        showlegend=True,
        width=700,
        height=500,
    )

    # Add mean lines
    fig.add_vline(
        x=true_vals.mean(),
        line_dash="dash",
        line_color="darkgreen",
        annotation_text=f"True mean: {true_vals.mean():.3f}",
    )

    fig.add_vline(
        x=false_vals.mean(),
        line_dash="dash",
        line_color="darkred",
        annotation_text=f"False mean: {false_vals.mean():.3f}",
    )

    return fig


def plot_aggregated_attention_flow(
    patterns: list,
    aggregation: str = "mean",
    title: str = "Aggregated Attention Flow",
) -> go.Figure:
    """Plot aggregated attention scores across all heads and layers.

    Args:
        patterns: List of AttentionPattern objects
        aggregation: How to aggregate ('mean', 'max', 'median')
        title: Plot title

    Returns:
        Plotly figure
    """
    # Assuming all patterns have the same structure
    n_layers = patterns[0].attention.shape[0]
    n_heads = patterns[0].attention.shape[1]

    # Compute aggregated scores for each pattern
    all_scores = []

    for pattern in patterns:
        # Get attention from prediction to all positions
        pred_pos = pattern.prediction_position
        # attention: [n_layers, n_heads, seq_len]
        attention_from_pred = pattern.attention[:, :, pred_pos, :]

        # Average across sequence
        avg_attention = attention_from_pred.mean(dim=-1)  # [n_layers, n_heads]
        all_scores.append(avg_attention)

    # Stack and aggregate
    all_scores = torch.stack(all_scores, dim=0)  # [n_patterns, n_layers, n_heads]

    if aggregation == "mean":
        aggregated = all_scores.mean(dim=0)
    elif aggregation == "max":
        aggregated = all_scores.max(dim=0).values
    elif aggregation == "median":
        aggregated = all_scores.median(dim=0).values
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=aggregated.numpy(),
            x=[f"H{h}" for h in range(n_heads)],
            y=[f"L{l}" for l in range(n_layers)],
            colorscale="Viridis",
            colorbar={"title": f"{aggregation.title()} Attention"},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Head",
        yaxis_title="Layer",
        width=max(800, n_heads * 40),
        height=max(600, n_layers * 40),
    )

    return fig


def plot_top_heads_comparison(
    significant_heads: list,
    true_scores_mean: torch.Tensor,
    false_scores_mean: torch.Tensor,
    top_k: int = 10,
) -> go.Figure:
    """Plot comparison of top heads' attention scores.

    Args:
        significant_heads: List of HeadScore objects
        true_scores_mean: Mean scores for true facts [n_layers, n_heads]
        false_scores_mean: Mean scores for false facts [n_layers, n_heads]
        top_k: Number of top heads to show

    Returns:
        Plotly bar chart
    """
    top_heads = significant_heads[:top_k]

    head_labels = [f"L{h.layer}H{h.head}" for h in top_heads]
    true_vals = [true_scores_mean[h.layer, h.head].item() for h in top_heads]
    false_vals = [false_scores_mean[h.layer, h.head].item() for h in top_heads]
    p_values = [h.p_value for h in top_heads]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="True Facts",
            x=head_labels,
            y=true_vals,
            marker_color="green",
            text=[f"{v:.3f}" for v in true_vals],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name="False Facts",
            x=head_labels,
            y=false_vals,
            marker_color="red",
            text=[f"{v:.3f}" for v in false_vals],
            textposition="auto",
        )
    )

    # Add p-value annotations
    annotations = []
    for i, (label, p_val) in enumerate(zip(head_labels, p_values)):
        annotations.append(
            {
                "x": label,
                "y": max(true_vals[i], false_vals[i]) * 1.1,
                "text": f"p={p_val:.2e}",
                "showarrow": False,
                "font": {"size": 9},
            }
        )

    fig.update_layout(
        title=f"Top {top_k} Factual Recall Heads",
        xaxis_title="Head",
        yaxis_title="Mean Attention to Subject",
        barmode="group",
        showlegend=True,
        annotations=annotations,
        width=max(800, top_k * 80),
        height=600,
    )

    return fig
