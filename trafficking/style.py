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
    if tdata is not None:
        phenos = sorted(tdata.obs["phenotype"].unique())
        if "phenotype_colors" in tdata.uns and len(tdata.uns["phenotype_colors"]) >= len(phenos):
            COLORS["tcell_phenotype"] = dict(zip(
                tdata.obs["phenotype"].cat.categories, tdata.uns["phenotype_colors"]))
        else:
            cols = tcell_cmap(np.linspace(0, 1, len(phenos)))
            COLORS["tcell_phenotype"] = {p: cols[i] for i, p in enumerate(phenos)}
    if mdata is not None:
        phenos = sorted(mdata.obs["phenotype"].unique())
        if "phenotype_colors" in mdata.uns and len(mdata.uns["phenotype_colors"]) >= len(phenos):
            COLORS["myeloid_phenotype"] = dict(zip(
                mdata.obs["phenotype"].cat.categories, mdata.uns["phenotype_colors"]))
        else:
            cols = myeloid_cmap(np.linspace(0.3, 0.9, len(phenos)))
            COLORS["myeloid_phenotype"] = {p: cols[i] for i, p in enumerate(phenos)}


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