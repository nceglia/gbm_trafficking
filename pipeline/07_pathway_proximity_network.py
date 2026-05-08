# %%
"""Pathway proximity networks from shared leading-edge genes.

For each (library, lineage), build a graph where nodes are pathways and edges
connect pairs sharing ≥ MIN_SHARED_GENES leading-edge genes.  Singletons are
dropped.  The full graph (all components together) is plotted on a single axes
with spring layout — connected components naturally separate.

Node encoding modes (--node-color):
  pie    — pie chart showing gene-level tissue affinity (default)
  family — solid color by pathway family


Usage (CLI):
    python pipeline/09_pathway_proximity_network.py
    python pipeline/09_pathway_proximity_network.py --node-color family
    python pipeline/09_pathway_proximity_network.py \
        --min-shared-genes 8 --k-constant 3.0 --libraries kegg
"""
import argparse
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.constants import DIRECTED_TISSUE_PAIRS
from modules.style import (
    PATHWAY_FAMILY_COLORS, PATHWAY_FAMILY_ORDER, EXCLUDED_FAMILY,
    TISSUE_COLORS,
)

# ── Defaults (overridable via CLI) ────────────────────────────────────
LIBRARIES = ("MSigDB_Hallmark_2020", "KEGG_2021_Human")
LIBRARY_SLUGS = {
    "MSigDB_Hallmark_2020": "hallmark",
    "KEGG_2021_Human": "kegg",
}
SLUG_TO_LIB = {v: k for k, v in LIBRARY_SLUGS.items()}

MIN_SHARED_GENES = 5
FDR_THRESHOLD = 0.25
MAX_NES_THRESHOLD = 1.5
K_CONSTANT = 2.0
N_EDGE_GENES = 3
SEED = 42
LAYOUT_SCALE = 1.0
NODE_COLOR_MODE = "pie"

NODE_SCALE = 0.5
NODE_SIZE_RANGE = (30, 300)
EDGE_WIDTH_RANGE = (0.5, 6.0)
PIE_RADIUS_RANGE = (0.01, 0.03)

LINEAGES = ("CD8", "CD4")
TISSUES = ("PBMC", "CSF", "TP")
INPUT_DIR = REPO_ROOT / "results" / "04_pseudobulk_de_gsea"
MODULES_DIR = REPO_ROOT / "pipeline" / "modules"
OUTPUT_DIR = INPUT_DIR / "networks"


def parse_args():
    p = argparse.ArgumentParser(
        description="Pathway proximity networks from shared leading-edge genes",
    )
    p.add_argument("--min-shared-genes", type=int, default=MIN_SHARED_GENES)
    p.add_argument("--k-constant", type=float, default=K_CONSTANT)
    p.add_argument("--n-edge-genes", type=int, default=N_EDGE_GENES)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--libraries", nargs="+", default=["hallmark", "kegg"],
                   choices=["hallmark", "kegg"])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--node-scale", type=float, default=NODE_SCALE,
                   help="Multiplier for node/pie size (default 0.5). "
                        "Smaller values = smaller circles.")
    p.add_argument("--layout-scale", type=float, default=LAYOUT_SCALE,
                   help="Scale factor for spring_layout (default 1.0).")
    p.add_argument("--split-components", action="store_true", default=False,
                   help="One PNG per connected component (deep-dive mode).")
    p.add_argument("--node-color", choices=["pie", "family"],
                   default=NODE_COLOR_MODE,
                   help="Node encoding: pie (gene-level tissue affinity) "
                        "or family (pathway family color). Default: pie.")
    return p.parse_args()


# ── Family map loader ─────────────────────────────────────────────────
def load_family_map(library_slug):
    path = MODULES_DIR / f"pathway_families_{library_slug}.tsv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df["term"], df["family"]))


# ── GSEA loading ──────────────────────────────────────────────────────
def load_gsea_long(library_name, library_tag):
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
    if not records:
        return pd.DataFrame()
    gsea_long = pd.concat(records, ignore_index=True)
    prefix = library_name + "__"
    gsea_long["pathway"] = gsea_long["Term"].str.replace(prefix, "", regex=False)
    return gsea_long


def filter_pathways(gsea_long, library_tag):
    """Return set of pathway names passing NES + FDR filters."""
    fmap = load_family_map(library_tag)
    stats = gsea_long.groupby(["lineage", "pathway"]).agg(
        max_abs_nes=("NES", lambda x: x.abs().max()),
        min_fdr=("FDR q-val", "min"),
    ).reset_index()
    passing = stats[
        (stats["max_abs_nes"] >= MAX_NES_THRESHOLD)
        & (stats["min_fdr"] <= FDR_THRESHOLD)
    ]
    pathways = set(passing["pathway"].unique())
    excluded = {pw for pw in pathways if fmap.get(pw) == EXCLUDED_FAMILY}
    return pathways - excluded


# ── Leading-edge gene collection ──────────────────────────────────────
def collect_lead_genes(gsea_long, lineage):
    lin = gsea_long[gsea_long["lineage"] == lineage]
    out = defaultdict(set)
    for _, row in lin.iterrows():
        pw = row["pathway"]
        lg = row.get("Lead_genes")
        if isinstance(lg, str) and lg:
            out[pw].update(g.strip() for g in lg.split(";") if g.strip())
    return dict(out)


# %% ── Cell 2: Build shared-gene graph ───────────────────────────────

def build_shared_gene_graph(pathways, lead_genes, min_shared_genes, library_tag):
    """Graph where edges = shared leading-edge genes >= threshold."""
    fmap = load_family_map(library_tag)
    G = nx.Graph()
    node_attrs = {}

    pw_list = sorted(pathways)
    for pw in pw_list:
        fam = fmap.get(pw)
        if fam and fam != EXCLUDED_FAMILY:
            G.add_node(pw)
            node_attrs[pw] = {"family": fam}

    pw_list = list(G.nodes())
    for i, j in combinations(range(len(pw_list)), 2):
        gi = lead_genes.get(pw_list[i], set())
        gj = lead_genes.get(pw_list[j], set())
        shared = gi & gj
        if len(shared) >= min_shared_genes:
            G.add_edge(
                pw_list[i], pw_list[j],
                shared_genes=len(shared),
                gene_list=tuple(sorted(shared)),
            )

    return G, node_attrs


def drop_singletons(G):
    """Remove nodes with degree 0 in-place, return count removed."""
    singletons = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(singletons)
    return len(singletons)


# %% ── Cell 3: Gene-level tissue affinity ─────────────────────────────

def load_de_stats(lineage):
    """Load DE Wald stats for a lineage. Returns {(t1, t2): {gene: stat}}."""
    de_stats = {}
    for t1, t2 in DIRECTED_TISSUE_PAIRS:
        path = INPUT_DIR / f"de_{lineage}_{t1}_vs_{t2}.csv"
        if not path.exists():
            print(f"  SKIP DE (not found): {path.name}")
            de_stats[(t1, t2)] = {}
            continue
        df = pd.read_csv(path, index_col=0)
        de_stats[(t1, t2)] = df["stat"].to_dict()
    return de_stats


def compute_gene_tissue_affinity(lead_genes, de_stats, pathways):
    """Compute per-pathway gene-level tissue fractions.

    Returns {pathway: {
        "frac_PBMC": float, "frac_CSF": float, "frac_TP": float,
        "genes_PBMC": [str], "genes_CSF": [str], "genes_TP": [str],
        "n_genes": int,
    }}
    """
    result = {}
    for pw in pathways:
        genes = lead_genes.get(pw)
        if not genes:
            continue

        genes_by_tissue = {"PBMC": [], "CSF": [], "TP": []}
        for g in sorted(genes):
            stat_pbmc_tp = de_stats.get(("PBMC", "TP"), {}).get(g, 0.0)
            stat_pbmc_csf = de_stats.get(("PBMC", "CSF"), {}).get(g, 0.0)
            stat_csf_tp = de_stats.get(("CSF", "TP"), {}).get(g, 0.0)

            aff_tp = np.mean([stat_pbmc_tp, stat_csf_tp])
            aff_pbmc = -np.mean([stat_pbmc_tp, stat_pbmc_csf])
            aff_csf = stat_pbmc_csf - stat_csf_tp

            affs = {"PBMC": aff_pbmc, "CSF": aff_csf, "TP": aff_tp}
            winner = max(affs, key=affs.get)
            genes_by_tissue[winner].append(g)

        n = len(genes)
        result[pw] = {
            "frac_PBMC": len(genes_by_tissue["PBMC"]) / n,
            "frac_CSF": len(genes_by_tissue["CSF"]) / n,
            "frac_TP": len(genes_by_tissue["TP"]) / n,
            "genes_PBMC": sorted(genes_by_tissue["PBMC"]),
            "genes_CSF": sorted(genes_by_tissue["CSF"]),
            "genes_TP": sorted(genes_by_tissue["TP"]),
            "n_genes": n,
        }
    return result


def compute_nes_coords(gsea_long, lineage, pathways):
    """Compute NES-derived ternary coordinates (z-score + softmax)."""
    lin = gsea_long[gsea_long["lineage"] == lineage]
    if len(lin) == 0:
        return {}

    affinities = {}
    for pw in pathways:
        pw_rows = lin[lin["pathway"] == pw]
        if len(pw_rows) == 0:
            continue
        tissue_aff = {}
        for tissue in TISSUES:
            up_vals, down_vals = [], []
            for t1, t2 in DIRECTED_TISSUE_PAIRS:
                label = f"{lineage}_{t1}_vs_{t2}"
                row = pw_rows[pw_rows["contrast"] == label]
                if len(row) == 0 or pd.isna(row.iloc[0]["NES"]):
                    continue
                nes = float(row.iloc[0]["NES"])
                if tissue == t2:
                    up_vals.append(nes)
                elif tissue == t1:
                    down_vals.append(nes)
            up_mean = np.mean(up_vals) if up_vals else 0.0
            down_mean = np.mean(down_vals) if down_vals else 0.0
            tissue_aff[tissue] = up_mean - down_mean
        affinities[pw] = tissue_aff

    if not affinities:
        return {}

    pw_list = sorted(affinities.keys())
    raw = np.array([[affinities[pw].get(t, 0.0) for t in TISSUES] for pw in pw_list])

    # z-score per tissue column
    for col in range(raw.shape[1]):
        mu, sd = raw[:, col].mean(), raw[:, col].std()
        if sd > 0:
            raw[:, col] = (raw[:, col] - mu) / sd
        else:
            raw[:, col] = 0.0

    # softmax per row
    coords = {}
    for i, pw in enumerate(pw_list):
        e = np.exp(raw[i])
        s = e / e.sum()
        coords[pw] = {"PBMC": float(s[0]), "CSF": float(s[1]), "TP": float(s[2])}
    return coords


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_concordance(gene_affinities, nes_coords):
    """Return {pathway: cosine_similarity}."""
    out = {}
    for pw in gene_affinities:
        if pw not in nes_coords:
            continue
        ga = gene_affinities[pw]
        nc = nes_coords[pw]
        gene_vec = [ga["frac_PBMC"], ga["frac_CSF"], ga["frac_TP"]]
        nes_vec = [nc["PBMC"], nc["CSF"], nc["TP"]]
        out[pw] = cosine_sim(gene_vec, nes_vec)
    return out


def write_affinity_csv(gene_affinities, nes_coords, concordance,
                       node_attrs, slug, lineage, output_dir):
    """Write gene_tissue_affinity_{slug}_{lineage}.csv."""
    rows = []
    for pw in gene_affinities:
        ga = gene_affinities[pw]
        nc = nes_coords.get(pw, {})
        rows.append({
            "pathway": pw,
            "family": node_attrs.get(pw, {}).get("family", ""),
            "n_leading_edge_genes": ga["n_genes"],
            "frac_PBMC": round(ga["frac_PBMC"], 4),
            "frac_CSF": round(ga["frac_CSF"], 4),
            "frac_TP": round(ga["frac_TP"], 4),
            "nes_coord_PBMC": round(nc.get("PBMC", 0), 4),
            "nes_coord_CSF": round(nc.get("CSF", 0), 4),
            "nes_coord_TP": round(nc.get("TP", 0), 4),
            "concordance_cosine": round(concordance.get(pw, 0), 4),
            "genes_PBMC": ";".join(ga["genes_PBMC"]),
            "genes_CSF": ";".join(ga["genes_CSF"]),
            "genes_TP": ";".join(ga["genes_TP"]),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("concordance_cosine", ascending=True).reset_index(drop=True)

    csv_path = output_dir / f"gene_tissue_affinity_{slug}_{lineage}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# %% ── Cell 4: Plotting ──────────────────────────────────────────────

def _scale_linear(values, lo, hi):
    arr = np.array(values, dtype=float)
    vmin, vmax = arr.min(), arr.max()
    if vmax == vmin:
        return np.full_like(arr, (lo + hi) / 2)
    return lo + (arr - vmin) / (vmax - vmin) * (hi - lo)


def _draw_pie_node(ax, x, y, fracs, colors, radius):
    theta_start = 90
    for frac, color in zip(fracs, colors):
        if frac <= 0:
            continue
        theta_end = theta_start - frac * 360
        wedge = Wedge((x, y), radius, theta_end, theta_start,
                      facecolor=color, edgecolor="black", linewidth=0.5)
        ax.add_patch(wedge)
        theta_start = theta_end


def _draw_nodes(ax, G, pos, node_attrs, gene_affinities, node_color_mode,
                node_scale=NODE_SCALE):
    """Draw nodes using the specified color mode. Returns node_list."""
    node_list = list(G.nodes())
    degrees = np.array([G.degree(n) for n in node_list], dtype=float)

    if node_color_mode == "pie" and gene_affinities:
        pie_colors = [TISSUE_COLORS["PBMC"], TISSUE_COLORS["CSF"], TISSUE_COLORS["TP"]]

        # compute radius range from layout extent, scaled by node_scale
        xs = [pos[n][0] for n in node_list]
        ys = [pos[n][1] for n in node_list]
        extent = max(max(xs) - min(xs), max(ys) - min(ys), 0.1)
        r_lo = extent * PIE_RADIUS_RANGE[0] * node_scale
        r_hi = extent * PIE_RADIUS_RANGE[1] * node_scale
        radii = _scale_linear(degrees, r_lo, r_hi)

        for i, n in enumerate(node_list):
            ga = gene_affinities.get(n)
            if ga:
                fracs = [ga["frac_PBMC"], ga["frac_CSF"], ga["frac_TP"]]
            else:
                fracs = [1.0 / 3, 1.0 / 3, 1.0 / 3]
            _draw_pie_node(ax, pos[n][0], pos[n][1], fracs, pie_colors, radii[i])
    else:
        node_colors = [
            PATHWAY_FAMILY_COLORS.get(
                node_attrs.get(n, {}).get("family"), "#bdc3c7")
            for n in node_list
        ]
        node_sizes = _scale_linear(
            degrees,
            NODE_SIZE_RANGE[0] * node_scale,
            NODE_SIZE_RANGE[1] * node_scale,
        )
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=node_list,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="k", linewidths=0.4,
        )

    return node_list


def _add_legend(ax, fig, node_list, node_attrs, node_color_mode):
    """Add appropriate legend for the node color mode."""
    if node_color_mode == "pie":
        pie_legend = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=TISSUE_COLORS[t],
                   markeredgecolor="k", markeredgewidth=0.4,
                   markersize=10, label=t)
            for t in TISSUES
        ]
        ax.legend(
            handles=pie_legend, loc="upper left",
            fontsize=8, title="Gene tissue affinity", title_fontsize=9,
            framealpha=0.9,
        )
    else:
        families_present = {
            node_attrs.get(n, {}).get("family") for n in node_list
        } - {None}
        legend_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PATHWAY_FAMILY_COLORS.get(fam, "#bdc3c7"),
                   markeredgecolor="k", markeredgewidth=0.4,
                   markersize=8, label=fam)
            for fam in PATHWAY_FAMILY_ORDER if fam in families_present
        ]
        if legend_handles:
            ax.legend(
                handles=legend_handles, loc="upper left",
                fontsize=7, title="Family", title_fontsize=8,
                framealpha=0.9,
            )


def _draw_edges_and_labels(ax, G, pos, n_edge_genes):
    """Draw edges, edge gene annotations, and node labels."""
    edge_shared = [G[u][v]["shared_genes"] for u, v in G.edges()]
    edge_widths = _scale_linear(edge_shared, EDGE_WIDTH_RANGE[0], EDGE_WIDTH_RANGE[1])

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color="#999", alpha=0.5,
    )

    if n_edge_genes > 0:
        for u, v, data in G.edges(data=True):
            genes = list(data["gene_list"])[:n_edge_genes]
            if not genes:
                continue
            mx = (pos[u][0] + pos[v][0]) / 2
            my = (pos[u][1] + pos[v][1]) / 2
            label = ", ".join(genes)
            if len(data["gene_list"]) > n_edge_genes:
                label += f" (+{len(data['gene_list']) - n_edge_genes})"
            ax.text(
                mx, my, label,
                fontsize=4.5, ha="center", va="center",
                color="#555", style="italic",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )

    node_list = list(G.nodes())
    if adjust_text is not None:
        texts = []
        for n in node_list:
            x, y = pos[n]
            texts.append(ax.text(x, y, n, fontsize=6, ha="center", va="center"))
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.5))
    else:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)


def _finish_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_network(G, node_attrs, gene_affinities, slug, lineage,
                 k_constant, n_edge_genes, seed, layout_scale,
                 node_color_mode, node_scale, output_dir):
    """Plot the full graph (all components together) on one axes."""
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return None

    scale = max(8, min(22, n_nodes * 0.4))
    fig, ax = plt.subplots(figsize=(scale, scale * 0.85))

    pos = nx.spring_layout(
        G, k=k_constant, seed=seed, iterations=200,
        weight="shared_genes", scale=layout_scale,
    )

    _draw_edges_and_labels(ax, G, pos, n_edge_genes)
    node_list = _draw_nodes(ax, G, pos, node_attrs, gene_affinities,
                            node_color_mode, node_scale)
    _add_legend(ax, fig, node_list, node_attrs, node_color_mode)

    n_components = nx.number_connected_components(G)
    ax.set_title(
        f"{slug} — {lineage}  "
        f"(n={n_nodes}, e={G.number_of_edges()}, "
        f"components={n_components})",
        fontsize=11, fontweight="bold",
    )
    _finish_ax(ax)
    ax.set_aspect("equal")

    fname = f"network_{slug}_{lineage}.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_component(comp, node_attrs, gene_affinities, slug, lineage, comp_idx,
                   k_constant, n_edge_genes, seed, layout_scale,
                   node_color_mode, node_scale, output_dir):
    """Plot a single connected component and save to output_dir."""
    n_nodes = comp.number_of_nodes()
    if n_nodes == 0:
        return None

    scale = max(6, min(18, n_nodes * 0.5))
    fig, ax = plt.subplots(figsize=(scale, scale * 0.85))

    pos = nx.spring_layout(
        comp, k=k_constant, seed=seed, iterations=200,
        weight="shared_genes", scale=layout_scale,
    )

    _draw_edges_and_labels(ax, comp, pos, n_edge_genes)
    node_list = _draw_nodes(ax, comp, pos, node_attrs, gene_affinities,
                            node_color_mode, node_scale)
    _add_legend(ax, fig, node_list, node_attrs, node_color_mode)

    ax.set_title(
        f"{slug} — {lineage} — component {comp_idx}  "
        f"(n={n_nodes}, e={comp.number_of_edges()})",
        fontsize=11, fontweight="bold",
    )
    _finish_ax(ax)
    ax.set_aspect("equal")

    fname = f"{slug}_{lineage}_component_{comp_idx:02d}.png"
    out_path = output_dir / fname
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Edge CSV export ───────────────────────────────────────────────────
def write_edge_csv(G, node_attrs, slug, lineage, output_dir):
    """Write edges_{slug}_{lineage}.csv with component rank and gene lists."""
    comp_rank = {}
    for rank, nodes in enumerate(
        sorted(nx.connected_components(G), key=len, reverse=True)
    ):
        for n in nodes:
            comp_rank[n] = rank

    rows = []
    for u, v, data in G.edges(data=True):
        pw1, pw2 = sorted([u, v])
        rows.append({
            "component_rank": comp_rank.get(pw1, -1),
            "pathway_1": pw1,
            "pathway_2": pw2,
            "family_1": node_attrs.get(pw1, {}).get("family", ""),
            "family_2": node_attrs.get(pw2, {}).get("family", ""),
            "n_shared_genes": data["shared_genes"],
            "shared_genes": ";".join(data["gene_list"]),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(
            ["component_rank", "n_shared_genes"],
            ascending=[True, False],
        ).reset_index(drop=True)

    csv_path = output_dir / f"edges_{slug}_{lineage}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, len(df)


# %% ── Cell 5: Main execution ────────────────────────────────────────

def run(args):
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_diagnostics = []
    all_saved = []
    all_concordance = {}

    lib_names = [SLUG_TO_LIB[s] for s in args.libraries]

    # Pre-load DE stats per lineage (shared across libraries)
    de_stats_by_lineage = {}
    for lineage in LINEAGES:
        de_stats_by_lineage[lineage] = load_de_stats(lineage)

    for lib_name in lib_names:
        slug = LIBRARY_SLUGS[lib_name]
        print(f"\n{'=' * 60}\nLoading {slug}...")
        gsea_long = load_gsea_long(lib_name, slug)
        if len(gsea_long) == 0:
            print(f"  No CSVs found for {slug}, skipping.")
            continue

        passing_pathways = filter_pathways(gsea_long, slug)
        print(f"  {slug}: {len(passing_pathways)} pathways passed filters")

        for lineage in LINEAGES:
            lead_genes = collect_lead_genes(gsea_long, lineage)
            lin_pathways = passing_pathways & set(lead_genes.keys())

            G, node_attrs = build_shared_gene_graph(
                lin_pathways, lead_genes, args.min_shared_genes, slug,
            )

            # Gene-level tissue affinity
            gene_affinities = compute_gene_tissue_affinity(
                lead_genes, de_stats_by_lineage[lineage],
                set(G.nodes()),
            )
            for pw, ga in gene_affinities.items():
                if pw in node_attrs:
                    node_attrs[pw].update({
                        "gene_frac_PBMC": ga["frac_PBMC"],
                        "gene_frac_CSF": ga["frac_CSF"],
                        "gene_frac_TP": ga["frac_TP"],
                    })

            # NES-derived coords + concordance
            nes_coords = compute_nes_coords(gsea_long, lineage, set(G.nodes()))
            concordance = compute_concordance(gene_affinities, nes_coords)
            all_concordance[(slug, lineage)] = concordance

            # Write affinity CSV (before dropping singletons — want all pathways)
            aff_csv = write_affinity_csv(
                gene_affinities, nes_coords, concordance,
                node_attrs, slug, lineage, output_dir,
            )
            print(f"    wrote: {aff_csv.name}")

            n_singletons = drop_singletons(G)
            n_components = nx.number_connected_components(G)
            comp_sizes = sorted(
                [len(c) for c in nx.connected_components(G)], reverse=True,
            )

            diag = {
                "library": slug,
                "lineage": lineage,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "n_components": n_components,
                "component_sizes": comp_sizes,
                "singletons_dropped": n_singletons,
            }
            all_diagnostics.append(diag)
            print(f"  {slug} / {lineage}: "
                  f"nodes={diag['nodes']}, edges={diag['edges']}, "
                  f"components={n_components} (sizes: {comp_sizes}), "
                  f"singletons={n_singletons} (dropped)")

            csv_path, n_edges_csv = write_edge_csv(
                G, node_attrs, slug, lineage, output_dir,
            )
            print(f"    wrote: {csv_path.name} ({n_edges_csv} edges)")

            if args.split_components:
                components = sorted(
                    nx.connected_components(G), key=len, reverse=True,
                )
                for comp_idx, comp_nodes in enumerate(components):
                    if len(comp_nodes) < 2:
                        continue
                    comp = G.subgraph(comp_nodes).copy()
                    out_path = plot_component(
                        comp, node_attrs, gene_affinities,
                        slug, lineage, comp_idx,
                        k_constant=args.k_constant,
                        n_edge_genes=args.n_edge_genes,
                        seed=args.seed,
                        layout_scale=args.layout_scale,
                        node_color_mode=args.node_color,
                        node_scale=args.node_scale,
                        output_dir=output_dir,
                    )
                    if out_path:
                        all_saved.append(str(out_path))
                        print(f"    saved: {out_path.name} "
                              f"(n={comp.number_of_nodes()}, "
                              f"e={comp.number_of_edges()})")
            else:
                out_path = plot_network(
                    G, node_attrs, gene_affinities,
                    slug, lineage,
                    k_constant=args.k_constant,
                    n_edge_genes=args.n_edge_genes,
                    seed=args.seed,
                    layout_scale=args.layout_scale,
                    node_color_mode=args.node_color,
                    node_scale=args.node_scale,
                    output_dir=output_dir,
                )
                if out_path:
                    all_saved.append(str(out_path))
                    print(f"    saved: {out_path.name}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("DIAGNOSTICS")
    print(f"{'=' * 60}")

    print("\nPer-(library, lineage) summary:")
    print(f"  {'library':<10} {'lineage':<6} {'nodes':>6} {'edges':>6} "
          f"{'comps':>6} {'dropped':>8} {'comp_sizes'}")
    for d in all_diagnostics:
        print(f"  {d['library']:<10} {d['lineage']:<6} "
              f"{d['nodes']:>6} {d['edges']:>6} "
              f"{d['n_components']:>6} {d['singletons_dropped']:>8}   "
              f"{d['component_sizes']}")

    # ── Concordance diagnostic ────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Gene-level vs NES-derived concordance")
    print(f"{'=' * 60}")
    for (slug, lineage), conc in sorted(all_concordance.items()):
        if not conc:
            continue
        vals = list(conc.values())
        print(f"\n  {slug} / {lineage}: "
              f"median cosine = {np.median(vals):.2f}, "
              f"mean = {np.mean(vals):.2f}, "
              f"min = {np.min(vals):.2f}")
        discordant = {pw: c for pw, c in conc.items() if c < 0.5}
        if discordant:
            print(f"  Discordant pathways (cosine < 0.5):")
            for pw in sorted(discordant, key=discordant.get):
                print(f"    {pw}: {discordant[pw]:.3f}")
        else:
            print("  None — gene-level and NES-derived affinities agree "
                  "across all pathways.")

    print(f"\nSaved {len(all_saved)} PNGs to {output_dir}/")
    print("Done.")


# %% ── Cell 6: Entry point ───────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run(args)
else:
    class _DefaultArgs:
        min_shared_genes = MIN_SHARED_GENES
        k_constant = K_CONSTANT
        n_edge_genes = N_EDGE_GENES
        output_dir = OUTPUT_DIR
        libraries = ["hallmark", "kegg"]
        seed = SEED
        node_scale = NODE_SCALE
        layout_scale = LAYOUT_SCALE
        split_components = False
        node_color = NODE_COLOR_MODE

    args = _DefaultArgs()
    run(args)

# %%
