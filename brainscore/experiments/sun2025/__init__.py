"""Sun 2025 motion task and analysis; source data lives in brainscore.data.sun2025."""
from .experiment import make_protocol
from .analysis import analyze

__all__ = ['make_protocol', 'analyze']
