"""Trajectory plot: CSF resident → CSF leaver → TP arriver → TP resident."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "pathway_motility_traffic"

agg = pd.read_csv(OUT_DIR / "motility_scores_per_edge.csv")

# Step labels along the CSF→TP trajectory
STEPS = [
    ("CSF resident",          "CSF_to_CSF", "source"),
    ("CSF leaver→TP",          "CSF_to_TP",  "source"),
    ("TP arriver←CSF",         "CSF_to_TP",  "target"),
    ("TP resident",            "TP_to_TP",   "source"),
]
# And the reverse leg
STEPS_REV = [
    ("TP resident",            "TP_to_TP",   "source"),
    ("TP leaver→CSF",          "TP_to_CSF",  "source"),
    ("CSF arriver←TP",         "TP_to_CSF",  "target"),
    ("CSF resident",           "CSF_to_CSF", "source"),
]

TERMS = [
    "KEGG: Cell adhesion molecules",
    "KEGG: Leukocyte transendothelial migration",
    "KEGG: Focal adhesion",
    "GOBP: Lymphocyte Migration (GO:0072676)",
    "GOBP: Lymphocyte Migration Into Lymphoid Organs (GO:0097021)",
    "GOBP: Lymphocyte Chemotaxis (GO:0048247)",
    "GOBP: T Cell Chemotaxis (GO:0010818)",
    "GOBP: Rho Protein Signal Transduction (GO:0007266)",
    "GOBP: Integrin-Mediated Signaling Pathway (GO:0007229)",
    "GOBP: Negative Regulation Of Actin Filament Polymerization (GO:0030837)",
    "GOBP: Leukocyte Chemotaxis Involved In Inflammatory Response (GO:0002232)",
    "Hallmark: Apical Junction",
]


def trajectory_values(lineage, term, steps):
    out = []
    for label, edge, role in steps:
        row = agg[(agg["lineage"] == lineage) & (agg["edge_label"] == edge)
                  & (agg["role"] == role) & (agg["pathway"] == term)]
        out.append(float(row["mean_score"].iloc[0]) if len(row) else np.nan)
    return out


for lineage in ["CD8", "CD4"]:
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=False)
    # Forward CSF→TP
    ax = axes[0]
    for term in TERMS:
        vals = trajectory_values(lineage, term, STEPS)
        ax.plot(range(len(STEPS)), vals, "o-", label=term.replace("GOBP: ", "").replace(" (GO:", " (")[:55])
    ax.set_xticks(range(len(STEPS)))
    ax.set_xticklabels([s[0] for s in STEPS])
    ax.set_ylabel("Mean motility-pathway score (log1p)")
    ax.set_title(f"{lineage}: motility trajectory CSF → TP")
    ax.axhline(0, color="grey", lw=0.5, linestyle="--")
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=7)
    # Reverse TP→CSF
    ax = axes[1]
    for term in TERMS:
        vals = trajectory_values(lineage, term, STEPS_REV)
        ax.plot(range(len(STEPS_REV)), vals, "o-", label=term.replace("GOBP: ", "").replace(" (GO:", " (")[:55])
    ax.set_xticks(range(len(STEPS_REV)))
    ax.set_xticklabels([s[0] for s in STEPS_REV])
    ax.set_ylabel("Mean motility-pathway score (log1p)")
    ax.set_title(f"{lineage}: motility trajectory TP → CSF (reverse)")
    ax.axhline(0, color="grey", lw=0.5, linestyle="--")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"motility_trajectory_{lineage}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"motility_trajectory_{lineage}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote motility_trajectory_{lineage}.png")
