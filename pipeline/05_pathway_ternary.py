#%%
"""Pathway ternary plot from clone pseudobulk GSEA results.

Each dot is one pathway, positioned by where its enrichment
lives in tissue space (PBMC / CSF / TP).

Taxonomy note
-------------
Hallmark and KEGG share a unified 14-family taxonomy defined in
``modules.style.PATHWAY_FAMILY_ORDER / PATHWAY_FAMILY_COLORS``.
GO BP uses a **separate** GO-slim-derived taxonomy with its own
color palette (generated at runtime and cached in
``pathway_families_gobp_colors.tsv``). This is intentional: GO slim
families don't map onto the manually curated 14-family scheme, so
each system gets its own legend.
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import rankdata

try:
    from adjustText import adjust_text
    print("adjustText installed")
    _HAS_ADJUST_TEXT = True
except ImportError:
    _HAS_ADJUST_TEXT = False
    print("WARNING: adjustText not installed; labels may overlap. "
          "Run: pip install adjustText")

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.celltyping_geometry import bary_to_cart, draw_simplex
from modules.style import (
    TISSUE_COLORS, PATHWAY_FAMILY_COLORS, PATHWAY_FAMILY_ORDER, EXCLUDED_FAMILY,
)
from modules.constants import DIRECTED_TISSUE_PAIRS
#%%
# ---- Library config ----
GENE_SETS = "MSigDB_Hallmark_2020"
FDR_THRESHOLD = 0.25
METHODS_TO_RUN = ("zscore",)  # add "softmax", "rank" back if needed

# Label config examples -- pass any of these to plot_ternary(label_config=...)
#
#   LabelConfig(mode="off")                       # no labels
#   LabelConfig(mode="top_n", top_n=10)           # 10 highest-|NES| overall
#   LabelConfig(mode="top_n_per_family", top_n=3) # 3 per family (default)
#   LabelConfig(mode="filter",
#               families=("Proliferation / cell cycle", "DNA damage / repair"))
#   LabelConfig(mode="filter", keywords=("TCR", "antigen"))
#   LabelConfig(mode="filter",
#               families=("NF-kB / Inflammation",), keywords=("TNF",))
@dataclass
class LabelConfig:
    mode: str = "top_n_per_family"  # "off", "top_n", "top_n_per_family", "filter"
    top_n: int = 3
    families: tuple = ()
    keywords: tuple = ()


LABEL_CONFIG_PER_LIBRARY = {
    "hallmark": LabelConfig(mode="top_n_per_family", top_n=3),
    "kegg":     LabelConfig(mode="top_n_per_family", top_n=2),
    "gobp":     LabelConfig(mode="off"),
}

LIBRARY_SLUGS = {
    "MSigDB_Hallmark_2020": "hallmark",
    "KEGG_2021_Human": "kegg",
    "GO_Biological_Process_2023": "gobp",
}
LIBRARY_TAG = LIBRARY_SLUGS[GENE_SETS]

MODULES_DIR = REPO_ROOT / "pipeline" / "modules"


def load_family_map(library_slug):
    """Read pipeline/modules/pathway_families_{slug}.tsv → {term: family}.

    Handles both 2-column (term, family) and 3-column (term, go_id, family) TSVs.
    """
    path = MODULES_DIR / f"pathway_families_{library_slug}.tsv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df["term"], df["family"]))


def load_or_create_gobp_colors(families):
    """Load or generate a stable color mapping for GO BP families.

    Reads from pathway_families_gobp_colors.tsv if it exists (preserving
    color stability across re-runs). For new families not in the file,
    assigns colors from tab20 + tab20b.  Writes back the full mapping.
    """
    import matplotlib.cm as cm

    tsv_path = MODULES_DIR / "pathway_families_gobp_colors.tsv"
    existing = {}
    if tsv_path.exists():
        df = pd.read_csv(tsv_path, sep="\t")
        existing = dict(zip(df["family"], df["color"]))

    # Assign new colors for unseen families
    used_colors = set(existing.values())
    palettes = [cm.tab20, cm.tab20b, cm.tab20c, cm.Set3]
    pool = []
    for pal in palettes:
        for i in range(pal.N):
            c = "#{:02x}{:02x}{:02x}".format(
                *[int(x * 255) for x in pal(i)[:3]]
            )
            if c not in used_colors:
                pool.append(c)

    color_map = dict(existing)
    pool_idx = 0
    for fam in sorted(families):
        if fam not in color_map:
            if pool_idx < len(pool):
                color_map[fam] = pool[pool_idx]
                pool_idx += 1
            else:
                color_map[fam] = "#bdc3c7"

    # Write back
    with open(tsv_path, "w") as f:
        f.write("family\tcolor\n")
        for fam in sorted(color_map.keys()):
            f.write(f"{fam}\t{color_map[fam]}\n")

    return color_map


def assign_family(pathway, library_tag):
    """Assign a pathway family from the TSV map. Returns None for unmapped terms."""
    fmap = _FAMILY_MAPS.get(library_tag)
    if fmap is None:
        return None  # no family map for this library yet
    return fmap.get(pathway)  # None if not in map


# Pre-load available family maps
_FAMILY_MAPS = {}
for _slug in LIBRARY_SLUGS.values():
    _m = load_family_map(_slug)
    if _m is not None:
        _FAMILY_MAPS[_slug] = _m


LINEAGES = ("CD8", "CD4")
TISSUES = ("PBMC", "CSF", "TP")
INPUT_DIR = REPO_ROOT / "results" / "04_pseudobulk_de_gsea"
NES_THRESHOLD = 1.5
TOP_LABELS = 8
# lower = more corner-pinned, higher = more centroid-clustered; tune if needed
TEMPERATURE = 1.0
METHOD = "zscore"  # one of "softmax", "rank", "zscore"
METHODS = ("softmax", "rank", "zscore")


#%%
# ---- Helper functions ----

def load_gsea_long(library_name, library_tag):
    """Load all 6 GSEA CSVs for a library into one long DataFrame."""
    records = []
    for lineage in LINEAGES:
        for t1, t2 in DIRECTED_TISSUE_PAIRS:
            label = f"{lineage}_{t1}_vs_{t2}"
            path = INPUT_DIR / f"gsea_{library_tag}_{label}.csv"
            if not path.exists():
                print(f"  SKIP (not found): {path.name}")
                continue
            gsea = pd.read_csv(path)
            gsea["lineage"] = lineage
            gsea["t1"] = t1
            gsea["t2"] = t2
            gsea["contrast"] = label
            records.append(gsea)
            print(f"  Loaded {path.name}: {len(gsea)} pathways")

    if not records:
        return pd.DataFrame()

    gsea_long = pd.concat(records, ignore_index=True)
    prefix = library_name + "__"
    gsea_long["pathway"] = gsea_long["Term"].str.replace(prefix, "", regex=False)
    return gsea_long


def build_affinity_df(gsea_long, library_tag):
    """Compute per-(lineage, pathway) tissue affinities from GSEA long table."""
    aff_records = []

    for lineage in LINEAGES:
        lin_df = gsea_long[gsea_long["lineage"] == lineage]
        if len(lin_df) == 0:
            continue
        nes_pivot = lin_df.pivot_table(
            index="pathway", columns="contrast", values="NES", aggfunc="first",
        )
        fdr_pivot = lin_df.pivot_table(
            index="pathway", columns="contrast", values="FDR q-val", aggfunc="first",
        )

        for pathway in lin_df["pathway"].unique():
            if pathway not in nes_pivot.index:
                continue
            row = nes_pivot.loc[pathway]
            fdr_row = fdr_pivot.loc[pathway] if pathway in fdr_pivot.index else None

            affinities = {}
            for tissue in TISSUES:
                up_vals, down_vals = [], []
                for t1, t2 in DIRECTED_TISSUE_PAIRS:
                    label = f"{lineage}_{t1}_vs_{t2}"
                    if label not in row.index or pd.isna(row[label]):
                        continue
                    nes = row[label]
                    if tissue == t2:
                        up_vals.append(nes)
                    elif tissue == t1:
                        down_vals.append(nes)
                up_mean = np.mean(up_vals) if up_vals else 0.0
                down_mean = np.mean(down_vals) if down_vals else 0.0
                affinities[tissue] = up_mean - down_mean

            all_nes = [row[f"{lineage}_{t1}_vs_{t2}"]
                       for t1, t2 in DIRECTED_TISSUE_PAIRS
                       if f"{lineage}_{t1}_vs_{t2}" in row.index
                       and pd.notna(row[f"{lineage}_{t1}_vs_{t2}"])]
            max_abs_nes = max(abs(v) for v in all_nes) if all_nes else 0.0

            all_fdr = []
            if fdr_row is not None:
                all_fdr = [fdr_row[f"{lineage}_{t1}_vs_{t2}"]
                           for t1, t2 in DIRECTED_TISSUE_PAIRS
                           if f"{lineage}_{t1}_vs_{t2}" in fdr_row.index
                           and pd.notna(fdr_row[f"{lineage}_{t1}_vs_{t2}"])]
            min_fdr = min(all_fdr) if all_fdr else 1.0

            aff_records.append({
                "lineage": lineage,
                "pathway": pathway,
                "PBMC": affinities["PBMC"],
                "CSF": affinities["CSF"],
                "TP": affinities["TP"],
                "max_abs_nes": max_abs_nes,
                "min_fdr": min_fdr,
                "family": assign_family(pathway, library_tag),
            })

    df = pd.DataFrame(aff_records)
    pass_nes = df["max_abs_nes"] >= NES_THRESHOLD
    pass_fdr = df["min_fdr"] <= FDR_THRESHOLD
    print(f"  Pathways total: {len(df)},  pass NES>={NES_THRESHOLD}: {pass_nes.sum()},  "
          f"pass FDR<={FDR_THRESHOLD}: {pass_fdr.sum()},  pass both: {(pass_nes & pass_fdr).sum()}")
    result = df[pass_nes & pass_fdr].reset_index(drop=True)

    # Strict check: after filtering, all surviving terms must have a family
    if library_tag in _FAMILY_MAPS:
        missing = result[result["family"].isna()]["pathway"].unique().tolist()
        if missing:
            raise KeyError(
                f"{len(missing)} filtered pathway(s) missing from "
                f"{library_tag} family map:\n"
                + "\n".join(f"  {t}" for t in sorted(missing))
                + f"\nUpdate pipeline/modules/pathway_families_{library_tag}.tsv"
            )
        # Exclude disease/infection sentinel family
        excluded_mask = result["family"] == EXCLUDED_FAMILY
        n_excluded = excluded_mask.sum()
        if n_excluded > 0:
            print(f"  Excluded {n_excluded} disease/infection pathways from "
                  f"{library_tag} plot")
            result = result[~excluded_mask].reset_index(drop=True)
    return result


def _softmax_row(x, t=TEMPERATURE):
    e = np.exp(np.asarray(x, dtype=float) / t)
    return e / e.sum()


def compute_coords(affinity_df, method):
    """Return a copy of affinity_df with PBMC/CSF/TP replaced by ternary coords."""
    df = affinity_df.copy()
    tissue_cols = list(TISSUES)

    if method == "softmax":
        coords = np.vstack(df[tissue_cols].apply(_softmax_row, axis=1).values)

    elif method == "rank":
        coords = np.vstack(
            df[tissue_cols].apply(
                lambda r: rankdata(r, method="average"), axis=1,
            ).values
        )
        coords = coords / coords.sum(axis=1, keepdims=True)

    elif method == "zscore":
        for lineage in df["lineage"].unique():
            mask = df["lineage"] == lineage
            for col in tissue_cols:
                vals = df.loc[mask, col]
                mu, sd = vals.mean(), vals.std()
                if sd > 0:
                    df.loc[mask, col] = (vals - mu) / sd
                else:
                    df.loc[mask, col] = 0.0
        coords = np.vstack(df[tissue_cols].apply(_softmax_row, axis=1).values)

    else:
        raise ValueError(f"Unknown method: {method}")

    df[tissue_cols] = coords
    return df


# %%
# ---- Cell 1: Load all 6 GSEA tables into one long DataFrame ----
gsea_long = load_gsea_long(GENE_SETS, LIBRARY_TAG)
print(f"\nCombined long DataFrame: {gsea_long.shape}")
if len(gsea_long) > 0:
    print(gsea_long.head())

# %%
# ---- Cell 2: Compute per-(lineage, pathway) ternary coordinates ----
all_coords = {}
affinity_df = pd.DataFrame()

if len(gsea_long) == 0:
    print(f"No GSEA CSVs found for {LIBRARY_TAG} — skipping cells 2-3.")
else:
    affinity_df = build_affinity_df(gsea_long, LIBRARY_TAG)
    print(f"\nAffinity DataFrame (after filters): {affinity_df.shape}")

    DIAG_PATHWAYS = ["TNF-alpha Signaling via NF-kB", "E2F Targets"]

    for method in METHODS:
        cdf = compute_coords(affinity_df, method)
        all_coords[method] = cdf

        print(f"\n--- {method} ---")
        cd8 = cdf[cdf["lineage"] == "CD8"]
        for pw in DIAG_PATHWAYS:
            hit = cd8[cd8["pathway"] == pw]
            if len(hit) == 0:
                print(f"  CD8 {pw}: not found (below threshold)")
            else:
                r = hit.iloc[0]
                print(f"  CD8 {pw}: PBMC={r['PBMC']:.2f}  CSF={r['CSF']:.2f}  TP={r['TP']:.2f}")

    print(f"\nPer lineage: {affinity_df.groupby('lineage').size().to_dict()}")
    print(f"Per family: {affinity_df['family'].value_counts().to_dict()}")

# %%
# ---- Cell 3: Ternary plots (one PNG per method, current library) ----

from matplotlib.lines import Line2D


def _select_label_rows(sub, label_config, has_families):
    """Return a DataFrame of rows to label, per label_config.mode."""
    if label_config.mode == "off" or len(sub) == 0:
        return sub.iloc[0:0]
    if label_config.mode == "top_n":
        return sub.nlargest(label_config.top_n, "max_abs_nes")
    if label_config.mode == "top_n_per_family":
        if not has_families or "family" not in sub.columns:
            return sub.nlargest(label_config.top_n, "max_abs_nes")
        parts = []
        for fam in sub["family"].dropna().unique():
            parts.append(
                sub[sub["family"] == fam].nlargest(label_config.top_n, "max_abs_nes")
            )
        return pd.concat(parts) if parts else sub.iloc[0:0]
    if label_config.mode == "filter":
        fam_mask = pd.Series(False, index=sub.index)
        kw_mask = pd.Series(False, index=sub.index)
        if label_config.families and has_families and "family" in sub.columns:
            fam_mask = sub["family"].isin(label_config.families)
        if label_config.keywords:
            kws_lower = [k.lower() for k in label_config.keywords]
            kw_mask = sub["pathway"].str.lower().apply(
                lambda p: any(k in p for k in kws_lower)
            )
        return sub[fam_mask | kw_mask]
    raise ValueError(f"Unknown label_config.mode: {label_config.mode}")


def _place_labels(ax, label_rows, fontsize):
    """Create text labels for label_rows and run adjust_text if available."""
    if len(label_rows) == 0:
        return
    texts = []
    for _, r in label_rows.iterrows():
        px, py = bary_to_cart(r["PBMC"], r["CSF"], r["TP"])
        t = ax.text(
            px, py, r["pathway"],
            fontsize=fontsize, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="none", alpha=0.7),
            zorder=5,
        )
        texts.append(t)
    if _HAS_ADJUST_TEXT and texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.6),
            expand_points=(4.3, 4.3),
            expand_text=(1.2, 1.2),
            force_text=(0.5, 0.8),
            force_points=(0.3, 0.5),
        )


def plot_ternary(coords_df, library_tag,
                 color_map=None, family_order=None, label_config=None,
                 savepath=None, title_tag=None):
    """Plot a ternary for the given coords DataFrame.

    Parameters
    ----------
    color_map : dict or None
        {family: hex_color}.  Defaults to PATHWAY_FAMILY_COLORS for
        Hallmark/KEGG; pass the GO-slim-derived dict for GOBP.
    family_order : tuple/list or None
        Legend ordering.  Defaults to PATHWAY_FAMILY_ORDER for
        Hallmark/KEGG.  For GOBP, pass only the families present in data.
    """
    has_families = library_tag in _FAMILY_MAPS
    if color_map is None:
        color_map = PATHWAY_FAMILY_COLORS
    if family_order is None:
        family_order = PATHWAY_FAMILY_ORDER
    if label_config is None:
        label_config = LabelConfig()
    if title_tag is None:
        title_tag = library_tag

    fig, axes = plt.subplots(1, 2, figsize=(14 / 3 * 2, 7 / 3 * 2))

    for ax, lineage in zip(axes, LINEAGES):
        draw_simplex(ax, lineage, ["PBMC", "CSF", "TP"])

        sub = coords_df[coords_df["lineage"] == lineage].copy()
        if len(sub) == 0:
            continue

        xc, yc = bary_to_cart(
            sub["PBMC"].values, sub["CSF"].values, sub["TP"].values,
        )

        sizes = (sub["max_abs_nes"].values ** 2) * 20
        if has_families:
            colors = [color_map.get(f, "#bdc3c7") for f in sub["family"]]
        else:
            colors = ["#bdc3c7"] * len(sub)

        ax.scatter(xc, yc, s=sizes, c=colors, alpha=0.7,
                   edgecolors="k", linewidths=0.4, zorder=3)

        label_rows = _select_label_rows(sub, label_config, has_families)
        _place_labels(ax, label_rows, fontsize=6)

        for t in ax.texts:
            txt = t.get_text()
            if txt in TISSUE_COLORS:
                t.set_color(TISSUE_COLORS[txt])

    # Legend
    legend_families = family_order
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=color_map.get(fam, "#bdc3c7"),
               markersize=8, markeredgecolor="k", markeredgewidth=0.4, label=fam)
        for fam in legend_families
    ]
    axes[1].legend(
        handles=legend_handles, loc="upper left",
        bbox_to_anchor=(1.02, 1.0), fontsize=7,
        title="Pathway family", title_fontsize=8, framealpha=0.9,
        borderaxespad=0,
    )

    plt.suptitle(
        f"Pathway Tissue Affinity — {title_tag}\n"
        "(clone pseudobulk DESeq2 → GSEA)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"Saved: {savepath}")
    plt.show()


if all_coords:
    if LIBRARY_TAG not in _FAMILY_MAPS:
        print(f"  Note: family mapping pending for {LIBRARY_TAG}. Dots colored gray.")

    # Choose color map and legend order for the current library
    if LIBRARY_TAG == "gobp" and LIBRARY_TAG in _FAMILY_MAPS:
        _gobp_families_in_data = sorted(
            affinity_df["family"].dropna().unique(),
            key=lambda f: affinity_df[affinity_df["family"] == f]["max_abs_nes"]
            .sum(),
            reverse=True,
        )
        _cur_color_map = load_or_create_gobp_colors(_gobp_families_in_data)
        _cur_family_order = _gobp_families_in_data
    else:
        _cur_color_map = PATHWAY_FAMILY_COLORS
        _cur_family_order = PATHWAY_FAMILY_ORDER

    for method in METHODS_TO_RUN:
        out_path = INPUT_DIR / f"pathway_ternary_{LIBRARY_TAG}_{method}.png"
        plot_ternary(all_coords[method], LIBRARY_TAG,
                     color_map=_cur_color_map,
                     family_order=_cur_family_order,
                     label_config=LABEL_CONFIG_PER_LIBRARY.get(
                         LIBRARY_TAG, LabelConfig()),
                     savepath=out_path,
                     title_tag=f"{LIBRARY_TAG} / {method}")

# %%
# ---- Cell 4: Combined plot — zscore method, all libraries, per lineage ----

ALL_LIBRARIES = {
    "MSigDB_Hallmark_2020": "hallmark",
    "KEGG_2021_Human": "kegg",
    "GO_Biological_Process_2023": "gobp",
}

# Load + compute zscore coords for every available library
lib_coords = {}
for lib_name, lib_tag in ALL_LIBRARIES.items():
    print(f"\n{'=' * 50}")
    print(f"Loading {lib_tag}...")
    gl = load_gsea_long(lib_name, lib_tag)
    if len(gl) == 0:
        print(f"  No CSVs found for {lib_tag}, skipping.")
        continue
    adf = build_affinity_df(gl, lib_tag)
    if len(adf) == 0:
        print(f"  No pathways passed filters for {lib_tag}, skipping.")
        continue
    lib_coords[lib_tag] = compute_coords(adf, "zscore")
    print(f"  {lib_tag}: {len(lib_coords[lib_tag])} pathways")

available_libs = list(lib_coords.keys())
n_libs = len(available_libs)

if n_libs == 0:
    print("No libraries with data — skipping combined plot.")
else:
    fig, axes = plt.subplots(
        n_libs, 2,
        figsize=(14 / 3 * 2, 3.5 * n_libs),
        squeeze=False,
    )

    # Build per-library color maps
    lib_color_maps = {}
    for lib_tag in available_libs:
        if lib_tag == "gobp" and lib_tag in _FAMILY_MAPS:
            cdf = lib_coords[lib_tag]
            gobp_fams = sorted(cdf["family"].dropna().unique())
            lib_color_maps[lib_tag] = load_or_create_gobp_colors(gobp_fams)
        else:
            lib_color_maps[lib_tag] = PATHWAY_FAMILY_COLORS

    for row_idx, lib_tag in enumerate(available_libs):
        cdf = lib_coords[lib_tag]
        has_families = lib_tag in _FAMILY_MAPS
        cmap = lib_color_maps[lib_tag]
        if not has_families:
            print(f"  {lib_tag}: no family map (gray dots)")
        for col_idx, lineage in enumerate(LINEAGES):
            ax = axes[row_idx, col_idx]
            draw_simplex(ax, f"{lib_tag} — {lineage}", ["PBMC", "CSF", "TP"])

            sub = cdf[cdf["lineage"] == lineage].copy()
            if len(sub) == 0:
                continue

            xc, yc = bary_to_cart(
                sub["PBMC"].values, sub["CSF"].values, sub["TP"].values,
            )
            sizes = (sub["max_abs_nes"].values ** 2) * 20
            if has_families:
                colors = [cmap.get(f, "#bdc3c7") for f in sub["family"]]
            else:
                colors = ["#bdc3c7"] * len(sub)

            ax.scatter(xc, yc, s=sizes, c=colors, alpha=0.7,
                       edgecolors="k", linewidths=0.4, zorder=3)

            lib_cfg = LABEL_CONFIG_PER_LIBRARY.get(lib_tag, LabelConfig())
            label_rows = _select_label_rows(sub, lib_cfg, has_families)
            _place_labels(ax, label_rows, fontsize=5)

            for t in ax.texts:
                txt = t.get_text()
                if txt in TISSUE_COLORS:
                    t.set_color(TISSUE_COLORS[txt])

    # Hallmark/KEGG legend (uses 14-family order)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PATHWAY_FAMILY_COLORS[fam],
               markersize=8, markeredgecolor="k", markeredgewidth=0.4, label=fam)
        for fam in PATHWAY_FAMILY_ORDER
    ]
    axes[0, 1].legend(
        handles=legend_handles, loc="upper left",
        bbox_to_anchor=(1.02, 1.0), fontsize=6,
        title="Pathway family", title_fontsize=7, framealpha=0.9,
        borderaxespad=0,
    )

    plt.suptitle(
        "Pathway Tissue Affinity — zscore, all libraries\n"
        "(clone pseudobulk DESeq2 → GSEA)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    out_path = INPUT_DIR / "pathway_ternary_all_libraries_zscore.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.show()

print("\nDone.")

# %%
# ---- Cell 5: Demo — Hallmark filter mode (proliferation + DNA repair) ----
if "hallmark" in _FAMILY_MAPS:
    _demo_long = load_gsea_long("MSigDB_Hallmark_2020", "hallmark")
    if len(_demo_long) > 0:
        _demo_adf = build_affinity_df(_demo_long, "hallmark")
        _demo_coords = compute_coords(_demo_adf, "zscore")
        _demo_path = (
            INPUT_DIR
            / "pathway_ternary_hallmark_zscore_proliferation_only.png"
        )
        plot_ternary(
            _demo_coords,
            "hallmark",
            color_map=PATHWAY_FAMILY_COLORS,
            family_order=PATHWAY_FAMILY_ORDER,
            label_config=LabelConfig(
                mode="filter",
                families=("Proliferation / cell cycle", "DNA damage / repair"),
            ),
            savepath=_demo_path,
            title_tag="hallmark / zscore (proliferation + DNA repair)",
        )

# %%
kegg_long = load_gsea_long("KEGG_2021_Human", "kegg")
kegg_adf = build_affinity_df(kegg_long, "kegg")
kegg_coords = compute_coords(kegg_adf, "zscore")

# Label anything in the TCR family OR mentioning "T cell receptor" in the name
plot_ternary(
    kegg_coords,
    "kegg",
    label_config=LabelConfig(
        mode="filter",
        families=("Proliferation / cell cycle",),
        # keywords=("repair", "prolif", "DNA"),
    ),
)
           
# %%
    