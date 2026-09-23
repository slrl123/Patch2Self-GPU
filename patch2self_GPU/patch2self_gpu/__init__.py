"""GPU implementation of DIPY's Patch2Self denoising (PyTorch)."""

from .denoise import patch2self_gpu

__version__ = "1.0.0"

__all__ = ["patch2self_gpu", "__version__"]
