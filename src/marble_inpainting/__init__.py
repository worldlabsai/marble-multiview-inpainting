"""Local input preparation for Marble multiview inpainting."""

from importlib.metadata import PackageNotFoundError, version

from marble_inpainting.models import PrepareConfig
from marble_inpainting.pipeline import prepare_auto, prepare_manual
from marble_inpainting.scene_io import load_scene

try:
    __version__ = version("marble-multiview-inpainting")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.1.0"

__all__ = [
    "PrepareConfig",
    "__version__",
    "load_scene",
    "prepare_auto",
    "prepare_manual",
]
