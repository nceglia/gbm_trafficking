"""Common constants reused across pipeline scripts."""

from itertools import combinations

from .style import TISSUE_ORDER

TISSUES = TISSUE_ORDER
TISSUE_PAIRS = tuple(combinations(TISSUES, 2))
DIRECTED_TISSUE_PAIRS = (("PBMC", "TP"), ("PBMC", "CSF"), ("CSF", "TP"))
MIN_CELLS = 10

