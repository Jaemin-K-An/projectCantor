import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.v3_5_0._model import clean_residuals

__all__ = ["clean_residuals"]
