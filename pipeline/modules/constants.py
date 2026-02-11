"""Common constants reused across pipeline scripts."""

from itertools import combinations

TISSUES = ("PBMC", "CSF", "TP")
TISSUE_PAIRS = tuple(combinations(TISSUES, 2))
MIN_CELLS = 10

