"""Shim — moved to viewers/build/clone_network.py."""
import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "viewers/build/clone_network.py"
sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name="__main__")
