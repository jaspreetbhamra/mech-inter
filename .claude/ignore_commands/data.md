# Data Command

Create or modify data loading and preprocessing pipelines.

## Usage
```
/data <dataset_name> <description>
```

## What This Command Does
1. Creates dataset class in `src/data/`
2. Implements PyTorch Dataset interface
3. Adds data preprocessing utilities
4. Creates data loading helper functions

## Output Structure
```python
# src/data/<dataset_name>.py
"""<Dataset description>."""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CustomDataset(Dataset):
    """
    <Dataset description>.
    
    Args:
        data_path: Path to data file
        split: One of 'train', 'val', 'test'
        transform: Optional transform to apply
    """
    
    def __init__(
        self,
        data_path: Path,
        split: str = "train",
        transform: Optional[Any] = None
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self._load_data()
        
    def _load_data(self) -> None:
        """Load and preprocess data."""
        pass
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item at index."""
        return x, y

def create_dataloaders(
    data_path: Path,
    batch_size: int = 32,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders."""
    pass
```

## Token Efficiency
- For standard datasets, reference existing implementations
- Only show novel preprocessing steps
- Assume knowledge of PyTorch DataLoader patterns