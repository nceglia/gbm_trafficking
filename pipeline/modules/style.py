"""Shared color and label definitions for the GBM trafficking project.
All plotting code should import from here rather than redefining locally.
"""

# ── Tissues ────────────────────────────────────────────────────────────
TISSUE_COLORS = {
    "PBMC": "#f0bd00",
    "CSF":  "#cd442a",
    "TP":   "#7e9437",
}
TISSUE_LABELS = {
    "PBMC": "PBMC",
    "CSF":  "CSF",
    "TP":   "Tumor",
}
TISSUE_ORDER = ("PBMC", "CSF", "TP")

# ── Lineages ───────────────────────────────────────────────────────────
LINEAGE_COLORS = {
    "CD8": "#2166ac",
    "CD4": "#b2182b",
}
LINEAGE_ORDER = ("CD8", "CD4")

# ── T cell phenotypes ──────────────────────────────────────────────────
# Ordered: CD8 activated → CD8 quiescent → CD4
# Palette designed for max distinguishability in narrow stacked bars:
# CD8 → cool hues (blues / teals / greens / purples)
# CD4 → warm hues (yellows / oranges / reds / magentas)
TCELL_PHENOTYPE_COLORS = {
    # CD8 Activated (cool, saturated)
    "CD8_Activated_TEMRA":   "#08306b",  # navy
    "CD8_Activated_TEXeff":  "#41b6c4",  # cyan/teal
    "CD8_Activated_TEXterm": "#54278f",  # dark purple
    "CD8_Activated_TRM":     "#1b7837",  # forest green
    # CD8 Quiescent (cool, lighter)
    "CD8_Quiescent_Memory":  "#5e8ec1",  # mid blue
    "CD8_Quiescent_Naive":   "#a6cee3",  # pale blue
    "CD8_Quiescent_TEXprog": "#9e9ac8",  # light purple
    # CD4 (warm)
    "CD4_Naive_Memory":      "#ffd92f",  # yellow
    "CD4_Exhausted":         "#c51b8a",  # magenta
    "CD4_Treg":              "#fd8d3c",  # orange
    "CD4_Th1_polarized":     "#e31a1c",  # red
    "CD4_Th2_polarized":     "#a50026",  # deep red
}
TCELL_PHENOTYPE_LABELS = {
    "CD8_Activated_TEMRA":   "TEMRA",
    "CD8_Activated_TEXeff":  "TEXeff",
    "CD8_Activated_TEXterm": "TEXterm",
    "CD8_Activated_TRM":     "TRM",
    "CD8_Quiescent_Memory":  "Memory",
    "CD8_Quiescent_Naive":   "Naive",
    "CD8_Quiescent_TEXprog": "TEXprog",
    "CD4_Naive_Memory":      "Naive/Memory",
    "CD4_Exhausted":         "Exhausted",
    "CD4_Treg":              "Treg",
    "CD4_Th1_polarized":     "Th1",
    "CD4_Th2_polarized":     "Th2",
}
TCELL_PHENOTYPE_ORDER = tuple(TCELL_PHENOTYPE_COLORS.keys())

# ── Myeloid phenotypes (placeholders — fill in when myeloid pipeline lands) ──
MYELOID_PHENOTYPE_COLORS: dict = {}  # TODO: populate
MYELOID_PHENOTYPE_LABELS: dict = {}  # TODO: populate
MYELOID_PHENOTYPE_ORDER: tuple = ()  # TODO: populate

# ── Timepoints ─────────────────────────────────────────────────────────
# Sequential colormap — early to late
TIMEPOINT_COLORS = {
    "1": "#fde0dd",
    "2": "#fcc5c0",
    "3": "#fa9fb5",
    "4": "#f768a1",
    "5": "#c51b8a",
    "6": "#7a0177",
}
TIMEPOINT_LABELS = {tp: f"T{tp}" for tp in "123456"}
TIMEPOINT_ORDER = ("1", "2", "3", "4", "5", "6")

# ── Patients ───────────────────────────────────────────────────────────
PATIENT_COLORS = {
    "DFCI1": "#1b9e77",
    "DFCI2": "#d95f02",
    "DFCI3": "#7570b3",
    "DFCI4": "#e7298a",
    "DFCI5": "#66a61e",
    "MSK1":  "#e6ab02",
}
PATIENT_MARKERS = {
    "DFCI1": "o", "DFCI2": "s", "DFCI3": "^",
    "DFCI4": "D", "DFCI5": "v", "MSK1":  "P",
}
PATIENT_ORDER = ("DFCI1", "DFCI2", "DFCI3", "DFCI4", "DFCI5", "MSK1")

# ── Pathway families (unified taxonomy across libraries) ───────────────
PATHWAY_FAMILY_ORDER = (
    "NF-kB / Inflammation",
    "Stress response",
    "Apoptosis / cell death",
    "Proliferation / cell cycle",
    "DNA damage / repair",
    "Type I/II interferon",
    "Cytokine signaling",
    "TCR / antigen receptor",
    "Metabolism — oxidative",
    "Metabolism — biosynthetic",
    "Migration / adhesion",
    "Growth factor signaling",
    "Hormone response",
    "Tissue morphogenesis",
)

PATHWAY_FAMILY_COLORS = {
    "NF-kB / Inflammation":          "#e41a1c",
    "Stress response":               "#ff7f00",
    "Apoptosis / cell death":        "#984ea3",
    "Proliferation / cell cycle":    "#377eb8",
    "DNA damage / repair":           "#80b1d3",
    "Type I/II interferon":          "#4daf4a",
    "Cytokine signaling":            "#a6d854",
    "TCR / antigen receptor":        "#ffd92f",
    "Metabolism — oxidative":        "#8dd3c7",
    "Metabolism — biosynthetic":     "#bc80bd",
    "Migration / adhesion":          "#fb8072",
    "Growth factor signaling":       "#b15928",
    "Hormone response":              "#cab2d6",
    "Tissue morphogenesis":          "#6a3d9a",
}

EXCLUDED_FAMILY = "Disease / infection (excluded)"

# ── GO BP pathway families (separate palette, derived from GO slim) ────
# Colors assigned at runtime from filtered data and saved to
# pipeline/modules/pathway_families_gobp_colors.tsv for stability.
GOBP_FAMILY_COLORS: dict = {}  # populated at runtime by 07_pathway_ternary.py
