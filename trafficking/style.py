import numpy as np
import matplotlib.pyplot as plt

TISSUE_ORDER = ["Plasma", "CSF", "TP"]

COLORS = {
    "tissue": {
        "Plasma": "#ff7d78",
        "CSF": "#d8e5f7",
        "TP": "#e8e8e8",
    },
    "tissue_labels": {
        "Plasma": "Blood",
        "CSF": "CSF",
        "TP": "Tumor",
    },
    "lineage": {
        "T cell": "#8E44AD",
        "Myeloid": "#16A085",
    },
    "patient": {
        "DFCI1": "#E74C3C",
        "DFCI2": "#3498DB",
        "DFCI3": "#2ECC71",
        "DFCI4": "#F39C12",
        "DFCI5": "#9B59B6",
        "MSK1": "#1ABC9C",
    },
    "tcell_phenotype": {},
    "myeloid_phenotype": {},
}


def tissue_color(tis):
    return COLORS["tissue"].get(tis, "#999")


def tissue_label(tis):
    return COLORS["tissue_labels"].get(tis, tis)


def patient_color(pat):
    return COLORS["patient"].get(pat, "#999")


def phenotype_color(pheno, modality="tcell"):
    key = "tcell_phenotype" if modality == "tcell" else "myeloid_phenotype"
    return COLORS[key].get(pheno, "#999")


def init_phenotype_colors(tdata=None, mdata=None,
                          tcell_cmap=plt.cm.tab20, myeloid_cmap=plt.cm.tab20c):
    """Populate COLORS['tcell_phenotype'] / ['myeloid_phenotype'].

    Resolution order per phenotype:
      1. Centralized mappings from pipeline.modules.style
         (TCELL_PHENOTYPE_COLORS, MYELOID_PHENOTYPE_COLORS).
      2. Any prebuilt list in adata.uns['phenotype_colors'] (legacy).
      3. Colormap-derived fallback so unmapped phenotypes still get a
         color and downstream code never throws a KeyError.
    """
    central_t, central_m = {}, {}
    try:
        from pipeline.modules.style import (
            TCELL_PHENOTYPE_COLORS as _CT,
            MYELOID_PHENOTYPE_COLORS as _CM,
        )
        central_t = dict(_CT)
        central_m = dict(_CM)
    except Exception:
        pass

    def _assign(adata, central, cmap, key):
        phenos = sorted(adata.obs["phenotype"].astype(str).unique())
        legacy = {}
        col = adata.obs["phenotype"]
        if (
            "phenotype_colors" in adata.uns
            and len(adata.uns["phenotype_colors"]) >= len(phenos)
            and hasattr(col, "cat")
        ):
            legacy = dict(zip(col.cat.categories, adata.uns["phenotype_colors"]))
        out, unmapped = {}, []
        for p in phenos:
            if p in central:
                out[p] = central[p]
            elif p in legacy:
                out[p] = legacy[p]
            else:
                unmapped.append(p)
        if unmapped:
            cols = cmap(np.linspace(0, 1, max(len(unmapped), 1)))
            for i, p in enumerate(unmapped):
                out[p] = cols[i]
        COLORS[key] = out

    if tdata is not None:
        _assign(tdata, central_t, tcell_cmap, "tcell_phenotype")
    if mdata is not None:
        _assign(mdata, central_m, myeloid_cmap, "myeloid_phenotype")


def init_colors_from_dataset(dataset):
    init_phenotype_colors(dataset.tdata, dataset.mdata)


def get_phenotype_colormap(modality="tcell"):
    key = "tcell_phenotype" if modality == "tcell" else "myeloid_phenotype"
    return dict(COLORS[key])


def apply_style(ax, despine=True, grid_alpha=0.15):
    if despine:
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax.tick_params(length=0)
    if grid_alpha > 0:
        ax.grid(alpha=grid_alpha, axis="y")