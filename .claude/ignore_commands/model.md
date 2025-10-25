# Model Command

Create or modify ML model architectures.

## Usage
```
/model <model_type> <description>
```

Examples:
- `/model transformer Add self-attention with relative positional encoding`
- `/model cnn Create ResNet-18 baseline`

## What This Command Does
1. Creates model file in `src/models/`
2. Implements model class with proper PyTorch structure
3. Adds unit tests for model (shape checks, forward pass)
4. Updates model registry if it exists

## Output Structure
```python
# src/models/<model_name>.py
"""<Model description>."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

class Model(nn.Module):
    """
    <Model description>.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        ...
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        **kwargs
    ):
        super().__init__()
        # Initialize layers
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional attention mask of shape (batch, seq_len)
            
        Returns:
            Output tensor of shape (batch, seq_len, output_dim)
        """
        # Implementation
        return output
```

## Token Efficiency
- Show only the model class definition
- Reference PyTorch docs for standard layers
- Assume familiarity with common architectures (skip explanation of standard components)
- For modifications, show only the changed methods