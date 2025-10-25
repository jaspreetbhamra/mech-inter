# Analysis Command

Create analysis scripts for results, visualizations, and metrics.

## Usage
```
/analysis <analysis_type> <description>
```

## What This Command Does
1. Creates analysis script in `src/analysis/`
2. Generates appropriate visualizations
3. Computes relevant metrics and statistics
4. Exports results in readable format

## Output Structure
```python
# src/analysis/<analysis_name>.py
"""<Analysis description>."""

from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results(
    results_path: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Analyze experimental results.
    
    Args:
        results_path: Path to results file
        output_dir: Directory to save plots
        
    Returns:
        Dictionary of computed metrics
    """
    # Load results
    # Compute metrics
    # Generate plots
    # Return summary
    pass

def plot_metric(data: pd.DataFrame, metric: str, save_path: Path) -> None:
    """Create publication-quality plot."""
    pass
```

## Token Efficiency
- Reference matplotlib gallery for standard plots
- Show only custom visualization code
- Assume familiarity with pandas/numpy operations