# %%
"""Figure 3 — CD8 trafficking patterns + per-transition sankeys + gene
signature for the CSF→TP draining transition.

Layout (top-to-bottom):
  A. Cross-compartment T-cell traffic patterns, per tissue pair
                                              (full width)
  B. "Draining CSF"           sankey (CSF→TP)
  C. "Draining Blood"         sankey (PBMC→TP)
  D. "Tumor → Blood Recirc."  sankey (TP→PBMC)
  E. Gene-level rewiring signature for "Draining CSF" (CSF→TP)
                                              (full width)

This script is a *composer* — it reads the already-rendered upstream
PNGs and tiles them. Re-running any of:
  • pipeline/traffic_archetype_graphs.py
  • pipeline/traffic_bayesian_sankey.py
  • pipeline/traffic_drainage_rewiring.py
…and then re-running this script will refresh the corresponding panel
without recomputing the rest.

Reads:
  results/traffic_archetype_graphs/trafficking_heat_crosses_thr0.06.png
  results/traffic_bayesian_sankey/sankey_individual/sankey_CD8_<src>_to_<dst>.png
  results/traffic_drainage_rewiring/CSF_to_TP/CD8/panel.png

Writes:
  results/figure3/figure3.png
  results/figure3/figure3.pdf
"""
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths  # noqa: E402

# =====================================================================
# Inputs
# =====================================================================
# Top row: the 3-panel per-tissue-pair cross figure at the 0.06
# edge-frequency threshold (from the traffic_archetype_graphs.py sweep).
TRAFFIC_HEAT_IMG = (paths.RESULTS_DIR / "traffic_archetype_graphs"
                     / "trafficking_heat_crosses_thr0.06.png")

SANKEY_DIR = (paths.RESULTS_DIR / "traffic_bayesian_sankey"
               / "sankey_individual")
SANKEY_PANELS = [
    ("Draining CSF",                  "sankey_CD8_CSF_to_TP.png"),
    ("Draining Blood",                "sankey_CD8_PBMC_to_TP.png"),
    ("Tumor → Blood Recirculation",   "sankey_CD8_TP_to_PBMC.png"),
]

OT_IMG = (paths.RESULTS_DIR / "traffic_drainage_rewiring"
           / "CSF_to_TP" / "CD8" / "panel.png")

OUT_DIR = paths.RESULTS_DIR / "figure3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Figure layout. HEIGHT_RATIOS are sized so that each row's subplot
# matches its source-image aspect when rendered at FIG_W width:
#   row A (cross-traffic, aspect 3.0)     needs 17/3    = 5.67" tall
#   row B/C/D (sankey, aspect ≈ 1.24)     needs 5.67/1.24 ≈ 4.57" tall
#   row E (OT panel, aspect ≈ 2.14)       needs 17/2.14 ≈ 7.94" tall
# Re-render the OT panel via
#   python pipeline/traffic_drainage_rewiring.py
#     --transitions CSF_to_TP --lineage CD8
#     --max-genes 35 --delta-thr 1.5 --bar-h 2.2
#     --no-title --legend-size 10 --skip-perm
# to keep its native aspect close to ~2.1 and suppress duplicate title.
FIG_W = 17.0
HEIGHT_RATIOS = (5.67, 4.57, 7.0)   # cross-traffic / sankeys / OT
HSPACE = 0.10
WSPACE = 0.05

TITLE_FS  = 13
LABEL_FS  = 16  # panel letters
DPI = 220


def _add_panel(ax, img_path, *, panel_label=None, title=None,
                aspect="equal"):
    """Render `img_path` inside `ax`. aspect='equal' preserves the
    source aspect (may letterbox); aspect='auto' stretches the image to
    fill the subplot exactly — use the latter when you need every row
    to occupy the same visual width."""
    if not img_path.exists():
        ax.text(0.5, 0.5, f"missing:\n{img_path.name}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#a00")
        ax.set_axis_off()
    else:
        img = mpimg.imread(str(img_path))
        ax.imshow(img, aspect=aspect, interpolation="lanczos")
        ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=8)
    if panel_label:
        ax.text(-0.01, 1.03, panel_label, transform=ax.transAxes,
                 fontsize=LABEL_FS, fontweight="bold",
                 va="bottom", ha="left")


def main():
    fig_h = sum(HEIGHT_RATIOS) + 1.3  # title/margin allowance
    fig = plt.figure(figsize=(FIG_W, fig_h))

    gs = gridspec.GridSpec(
        3, 3,
        height_ratios=HEIGHT_RATIOS,
        hspace=HSPACE, wspace=WSPACE,
        left=0.03, right=0.99, top=0.96, bottom=0.02,
    )

    # Row A — cross-compartment traffic patterns (full width).
    # The source image already carries its own title + colorbar; we
    # only add the panel letter so we don't double-up.
    ax_a = fig.add_subplot(gs[0, :])
    _add_panel(ax_a, TRAFFIC_HEAT_IMG, panel_label="A")

    # Row B/C/D — three model sankeys
    panel_letters = ["B", "C", "D"]
    for i, (title, fname) in enumerate(SANKEY_PANELS):
        ax = fig.add_subplot(gs[1, i])
        _add_panel(ax, SANKEY_DIR / fname,
                    panel_label=panel_letters[i], title=title)

    # Row E — gene-level OT signature (full width).
    # The OT panel is now regenerated with --no-title (it only carries
    # its legend), so we set the single title here.
    ax_e = fig.add_subplot(gs[2, :])
    _add_panel(ax_e, OT_IMG, panel_label="E",
                title="Draining CSF  ·  Gene-level rewiring signature")

    fig.savefig(OUT_DIR / "figure3.png", dpi=DPI,
                 bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure3.pdf",
                 bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/figure3.png + figure3.pdf")


if __name__ == "__main__":
    main()
