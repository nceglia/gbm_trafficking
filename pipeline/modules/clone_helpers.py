"""Shared clone/lineage helpers."""


def infer_lineage_from_phenotype(phenotype):
    return "CD8" if "CD8" in str(phenotype) else "CD4"


def shorten_phenotype_label(phenotype):
    label = str(phenotype)
    return (
        label.replace("CD8_Activated_", "")
        .replace("CD8_Quiescent_", "")
        .replace("CD4_", "")
    )


def abbreviate_phenotype_label(phenotype):
    label = str(phenotype)
    return (
        label.replace("CD8_Activated_", "CD8a_")
        .replace("CD8_Quiescent_", "CD8q_")
    )


def parse_node_id(node_id, tissues, timepoints):
    for tissue in sorted(tissues, key=len, reverse=True):
        if node_id.startswith(tissue + "_"):
            timepoint = node_id[len(tissue) + 1 :]
            if timepoint in timepoints:
                return tissue, timepoint
    return None, None


def categorize_edge(source, target, tissues, timepoints):
    t1, tp1 = parse_node_id(source, tissues, timepoints)
    t2, tp2 = parse_node_id(target, tissues, timepoints)
    if t1 is None or t2 is None:
        return "other"
    if t1 == t2:
        return "within_tissue"
    if tp1 == tp2:
        return "cross_tissue_same_time"
    return "cross_tissue_diff_time"
