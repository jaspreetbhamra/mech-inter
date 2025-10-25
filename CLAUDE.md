# CLAUDE.md - ML Research Project Configuration

## Project Overview
**Project Code**: MLRESEARCH-INTERP-PHASE1
**Primary Language**: Python
**Focus**: Machine Learning Research - Interpretability Phase 1
**Environment Manager**: uv


## Import Style
**CRITICAL**: Use absolute imports from project root, never relative imports.

✅ CORRECT:
```python
from utils.tools import pca
from models.transformer import TransformerModel
from data.loader import DataLoader
```

❌ INCORRECT:
```python
from ..utils.tools import pca
from .tools import pca
```

## Environment Management
- Use `uv` for all dependency management
- Python version: 3.10+
- Virtual environment location: `.venv/`

## Code Standards

### Python Style
- Follow PEP 8
- Type hints required for all function signatures
- Docstrings: Google style for public APIs
- Max line length: 100 characters
- Use dataclasses for configuration objects

### ML-Specific Guidelines
1. **Reproducibility**: Always set random seeds (use `src.utils.seed.set_seed()`)
2. **Configs**: Use yaml/json configs, never hardcode hyperparameters
3. **Logging**: Use Python logging module, not print statements
4. **Experiments**: Track with MLflow/Weights & Biases
5. **Data**: Never commit data files, use data versioning (DVC)
6. **Cache**: Make sure to cache any long-running/expensive operations

### File Organization
- One model per file in `src/models/`
- One dataset class per file in `src/data/`
- Experiments in `src/experiments/`, named by date: `exp_2025_01_15_baseline.py`
- Utility functions in `src/utils/`, well-organized by purpose

## Token Efficiency Rules

### When to be Concise
1. **Read operations**: Only show relevant code sections, not entire files
2. **Imports**: Don't repeat standard library imports in responses
3. **Boilerplate**: Reference existing patterns, don't regenerate
4. **Comments**: Assume ML knowledge, explain only novel approaches

### When to be Verbose
1. **Novel algorithms**: Explain mathematical reasoning
2. **Complex architectures**: Show full model definitions
3. **Bug fixes**: Show before/after with explanation
4. **Experiments**: Full config and rationale

### Context Management
- When editing, show only the function/class being modified + 5 lines context
- For new files, generate complete implementation
- Reference existing files by path rather than showing their content
- Use descriptive variable names to minimize need for comments


## Testing Standards
- Unit tests for all utility functions
- Integration tests for data pipelines
- Model tests: shape checks, gradient flow, basic sanity checks
- Use `pytest` with fixtures for common setup
- Test file naming: `test_<module_name>.py`

## Git Workflow
- Branch naming: `feature/<description>`, `fix/<description>`, `exp/<experiment-name>`
- Commit messages: `<type>: <description>` (e.g., "feat: add transformer baseline")
- Types: feat, fix, exp, refactor, docs, test, perf
- Never commit: notebooks with outputs, large data files, model checkpoints, `.venv/`

## Documentation
- README.md: Project overview, setup instructions, basic usage
- Each module: Brief Google-style docstring at top explaining purpose
- Complex functions: Docstring with Args, Returns, Raises, Example
- Experiments: Document hypothesis, methodology, results in experiment file header

## Common Commands Reference
- Setup: `uv venv && source .venv/bin/activate && uv pip install -e .`
- Run experiment: `python -m src.experiments.exp_<name>`
- Tests: `pytest tests/`
- Formatting: `black src/ tests/ && ruff check src/ tests/`
- Type checking: `mypy src/`

## Notes for Claude
- Assume user has deep ML knowledge unless stated otherwise
- Focus responses on research insights and implementation details
- When suggesting experiments, provide full runnable code with configs
- For bugs, explain root cause and ML implications, not just fix
- Prioritize reproducibility and scientific rigor in all suggestions
- Don't run tests on every run
- Don't generate unnecessary files, if there are summary files that you think would be helpful, ask me to confirm if I want them