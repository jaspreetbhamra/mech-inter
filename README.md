# Mechanistic Interpretability Research Environment

A comprehensive toolkit for mechanistic interpretability research on transformer language models, built on [TransformerLens](https://github.com/neelnanda-io/TransformerLens).

## Features

- **Activation Patching**: Causal tracing and path patching for identifying important model components
- **Sparse Autoencoders**: Train SAEs to discover interpretable features in activations
- **Circuit Discovery**: Automated ablation, pruning, and circuit extraction
- **Visualization**: Interactive plots for attention patterns, activation heatmaps, and circuit graphs
- **Experiment Tracking**: Integrated W&B support with local logging fallback

## Installation

### Local Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Install in editable mode
uv pip install -e .
```

### Google Colab Setup

Open `experiments/colab_setup.ipynb` in Colab. It will:
1. Install all dependencies
2. Load a pre-trained model (GPT-2 or Llama)
3. Provide helper functions for activation extraction
4. Set up experiment tracking

## Quick Start

### Loading a Model

```python
from src.utils import load_model, set_seed

# Set random seed for reproducibility
set_seed(42)

# Load model
model = load_model("gpt2-medium")
```

### Activation Patching

```python
from src.activation_patching import comprehensive_activation_patching

# Define clean and corrupted prompts
clean = "The Eiffel Tower is located in the city of Paris"
corrupted = "The Eiffel Tower is located in the city of London"

# Answer tokens: [correct_id, incorrect_id]
answer_tokens = [3000, 3576]  # Example token IDs

# Run activation patching
results = comprehensive_activation_patching(
    model,
    clean_prompt=clean,
    corrupted_prompt=corrupted,
    answer_tokens=answer_tokens,
    components=["resid_post", "attn_out", "mlp_out"],
)

# Visualize results
from src.visualization import plot_activation_patching_results

fig = plot_activation_patching_results(
    results,
    n_layers=model.cfg.n_layers,
    components=["resid_post", "attn_out", "mlp_out"],
)
fig.show()
```

### Training a Sparse Autoencoder

```python
from src.sae_training import SparseAutoencoder, collect_activations, train_sae
from src.utils import load_model
from torch.utils.data import DataLoader, TensorDataset

# Load model
model = load_model("gpt2-medium")

# Collect MLP activations from your dataset
prompts = ["your", "prompts", "here"]
activations = collect_activations(
    model,
    prompts,
    layer=6,
    component="mlp_out",
)

# Create SAE
sae = SparseAutoencoder(
    d_model=model.cfg.d_model,
    d_hidden=model.cfg.d_model * 4,  # 4x expansion
    sparsity_coef=1e-3,
)

# Train
train_loader = DataLoader(TensorDataset(activations), batch_size=32, shuffle=True)
train_sae(sae, train_loader, n_epochs=10, lr=1e-3)

# Save
from src.sae_training import save_sae
save_sae(sae, "saes/gpt2_l6_mlp.pt", metadata={"layer": 6, "component": "mlp_out"})
```

### Circuit Discovery

```python
from src.circuit_discovery import iterative_pruning

# Find minimal circuit for a task
important_heads = iterative_pruning(
    model,
    prompt="The Eiffel Tower is located in",
    answer_tokens=[3000, 3576],
    threshold=0.01,  # Maximum effect to allow pruning
    ablation_type="mean",
)

print(f"Important heads: {important_heads}")

# Visualize circuit
from src.visualization import plot_circuit_graph

fig = plot_circuit_graph(
    important_heads,
    n_layers=model.cfg.n_layers,
    n_heads=model.cfg.n_heads,
)
fig.show()
```

### Experiment Tracking

```python
from src.experiment_tracker import track_experiment

# Use as context manager
with track_experiment(
    experiment_name="activation_patching_facts",
    config={"model": "gpt2-medium", "task": "fact_recall"},
    use_wandb=True,  # Set to False for local-only logging
) as tracker:

    for step, prompt in enumerate(dataset):
        # Your experiment code
        loss = run_experiment(prompt)

        # Log metrics
        tracker.log({"loss": loss, "accuracy": acc}, step=step)

    # Log figure
    tracker.log_figure("results", fig)
```

## Project Structure

```
mech-inter/
├── data/               # Fact datasets
├── models/             # Model checkpoints
├── saes/               # Trained sparse autoencoders
├── experiments/        # Jupyter notebooks
│   └── colab_setup.ipynb
├── src/
│   ├── activation_patching.py   # Causal tracing utilities
│   ├── sae_training.py          # SAE training and analysis
│   ├── circuit_discovery.py     # Ablation and pruning
│   ├── visualization.py         # Plotting utilities
│   ├── experiment_tracker.py    # W&B + local logging
│   └── utils.py                 # Model loading and helpers
├── results/            # Figures, logs
├── requirements.txt
└── README.md
```

## Common Hook Names (TransformerLens)

For extracting activations:

- `blocks.{l}.hook_resid_pre` - Residual stream before layer
- `blocks.{l}.hook_resid_mid` - After attention, before MLP
- `blocks.{l}.hook_resid_post` - After full layer
- `blocks.{l}.attn.hook_pattern` - Attention patterns [batch, head, query, key]
- `blocks.{l}.attn.hook_result` - Attention output per head [batch, seq, head, d_head]
- `blocks.{l}.attn.hook_q`, `hook_k`, `hook_v` - Query/Key/Value vectors
- `blocks.{l}.mlp.hook_pre` - MLP input (pre-activation)
- `blocks.{l}.mlp.hook_post` - MLP neuron activations (post-GELU)

## Key Concepts

### Activation Patching
Replace activations in a "clean" run with activations from a "corrupted" run to identify which components are causally important for a behavior.

### Sparse Autoencoders (SAEs)
Train autoencoders with sparsity constraints to discover interpretable features in model activations. The learned features often correspond to meaningful semantic concepts.

### Circuit Discovery
Use ablation and iterative pruning to identify the minimal set of components (attention heads, neurons) required for a specific task.

### Direct Effects
Measure how much a component's output directly influences the final logits, ignoring effects mediated through later layers.

## Supported Models

Via TransformerLens:
- GPT-2 (Small, Medium, Large, XL)
- GPT-Neo / GPT-J
- Llama 2 / Llama 3.2 (requires HF token)
- Pythia suite
- Any HuggingFace model compatible with TransformerLens

## Examples

Check `experiments/` directory for example notebooks:
- `colab_setup.ipynb` - Full setup guide with examples
- More examples coming soon!

## Resources

- [TransformerLens Documentation](https://transformerlens.readthedocs.io/)
- [Neel Nanda's Interpretability Glossary](https://www.neelnanda.io/mechanistic-interpretability/glossary)
- [Activation Patching Paper](https://arxiv.org/abs/2211.00593)
- [Sparse Autoencoders for MI](https://arxiv.org/abs/2309.08600)

## Development

```bash
# Run tests
pytest tests/

# Format code
black src/ tests/
ruff check src/ tests/

# Type checking
mypy src/
```

## Acknowledgments

Built on [TransformerLens](https://github.com/neelnanda-io/TransformerLens) by Neel Nanda and the interpretability research community.
