import numpy as np
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _clean_tcr(adata, lineage, clone_key, phenotype_key, tissue_key):
    tcr = adata.obs[adata.obs[clone_key].notna()].copy()
    tcr["_lin"] = tcr[phenotype_key].apply(lambda x: "CD8" if "CD8" in x else "CD4")
    # remove lineage-switching clones
    cl = tcr.groupby(clone_key)["_lin"].nunique()
    tcr = tcr[~tcr[clone_key].isin(cl[cl > 1].index)]
    tcr = tcr[tcr["_lin"] == lineage]
    tcr["pheno"] = (tcr[phenotype_key]
        .str.replace("CD8_Activated_", "", regex=False)
        .str.replace("CD8_Quiescent_", "", regex=False)
        .str.replace("CD4_", "", regex=False))
    return tcr


def _count_phenotypes(cells, phenotype_col, pidx, K):
    counts = np.zeros(K, dtype=int)
    for p, n in cells[phenotype_col].value_counts().items():
        if p in pidx:
            counts[pidx[p]] = n
    return counts


def extract_transitions(adata, t1, t2, lineage="CD8", clone_key="trb",
                        phenotype_key="phenotype", tissue_key="tissue",
                        patient_key="patient"):
    """Extract per-clone phenotype count vectors between two tissues, pooling all timepoints."""
    tcr = _clean_tcr(adata, lineage, clone_key, phenotype_key, tissue_key)

    clone_tissues = tcr.groupby(clone_key)[tissue_key].apply(set)
    shared = clone_tissues[clone_tissues.apply(lambda s: t1 in s and t2 in s)].index
    phenotypes = sorted(tcr["pheno"].unique())
    K = len(phenotypes)
    pidx = {p: i for i, p in enumerate(phenotypes)}

    records = []
    for clone in shared:
        cells = tcr[tcr[clone_key] == clone]
        src = _count_phenotypes(cells[cells[tissue_key] == t1], "pheno", pidx, K)
        dst = _count_phenotypes(cells[cells[tissue_key] == t2], "pheno", pidx, K)
        pat = cells[patient_key].iloc[0] if patient_key in cells.columns else "unknown"
        records.append({"clone": clone, "patient": pat, "src": src, "dst": dst,
                        "n_src": src.sum(), "n_dst": dst.sum()})

    return pd.DataFrame(records), phenotypes


def extract_temporal_transitions(adata, t1, t2, lineage="CD8", clone_key="trb",
                                 phenotype_key="phenotype", tissue_key="tissue",
                                 patient_key="patient", time_key="timepoint"):
    """Extract per-clone transitions between two tissues at consecutive timepoint pairs.

    A clone observed at timepoints [1, 3, 5] in both tissues generates two observations:
    (T1→T3) and (T3→T5). Source phenotype is from the earlier timepoint, destination
    from the later timepoint. This ensures all transitions are forward-in-time.
    """
    tcr = _clean_tcr(adata, lineage, clone_key, phenotype_key, tissue_key)
    tcr = tcr[tcr[tissue_key].isin([t1, t2])]

    phenotypes = sorted(tcr["pheno"].unique())
    K = len(phenotypes)
    pidx = {p: i for i, p in enumerate(phenotypes)}

    # get sorted timepoints
    all_timepoints = sorted(tcr[time_key].unique())
    tp_to_idx = {tp: i for i, tp in enumerate(all_timepoints)}

    # for each clone, find timepoints where it appears in BOTH tissues
    records = []
    for clone, grp in tcr.groupby(clone_key):
        pat = grp[patient_key].iloc[0] if patient_key in grp.columns else "unknown"
        # timepoints with cells in source tissue
        src_tps = set(grp[grp[tissue_key] == t1][time_key].unique())
        # timepoints with cells in destination tissue
        dst_tps = set(grp[grp[tissue_key] == t2][time_key].unique())
        # timepoints present in both
        both_tps = sorted(src_tps & dst_tps, key=lambda x: tp_to_idx.get(x, 0))

        if len(both_tps) < 2:
            # if clone appears at only 1 shared timepoint, use it as a single
            # pooled observation (same as non-temporal model)
            if len(both_tps) == 1:
                tp = both_tps[0]
                src_cells = grp[(grp[tissue_key] == t1) & (grp[time_key] == tp)]
                dst_cells = grp[(grp[tissue_key] == t2) & (grp[time_key] == tp)]
                src = _count_phenotypes(src_cells, "pheno", pidx, K)
                dst = _count_phenotypes(dst_cells, "pheno", pidx, K)
                if src.sum() > 0 and dst.sum() > 0:
                    records.append({"clone": clone, "patient": pat,
                                    "src": src, "dst": dst,
                                    "n_src": src.sum(), "n_dst": dst.sum(),
                                    "tp_src": tp, "tp_dst": tp, "temporal": False})
            continue

        # generate consecutive pairs
        for i in range(len(both_tps) - 1):
            tp_src, tp_dst = both_tps[i], both_tps[i + 1]
            src_cells = grp[(grp[tissue_key] == t1) & (grp[time_key] == tp_src)]
            dst_cells = grp[(grp[tissue_key] == t2) & (grp[time_key] == tp_dst)]
            src = _count_phenotypes(src_cells, "pheno", pidx, K)
            dst = _count_phenotypes(dst_cells, "pheno", pidx, K)
            if src.sum() > 0 and dst.sum() > 0:
                records.append({"clone": clone, "patient": pat,
                                "src": src, "dst": dst,
                                "n_src": src.sum(), "n_dst": dst.sum(),
                                "tp_src": tp_src, "tp_dst": tp_dst, "temporal": True})

    df = pd.DataFrame(records)
    if len(df) > 0:
        n_temporal = df["temporal"].sum()
        n_static = (~df["temporal"]).sum()
        print(f"  Temporal observations: {n_temporal}, Static (single-tp): {n_static}")
    return df, phenotypes


def prepare_tensors(data):
    """Convert DataFrame of clone observations to torch tensors for Pyro."""
    src = torch.tensor(np.stack(data["src"].values), dtype=torch.float, device=device)
    theta = (src + 0.1)
    theta = theta / theta.sum(dim=1, keepdim=True)
    dst = torch.tensor(np.stack(data["dst"].values), dtype=torch.float, device=device)
    n_dst = torch.tensor(data["n_dst"].values, dtype=torch.float, device=device)
    pat_ids, pat_names = pd.factorize(data["patient"].values)
    pat_ids = torch.tensor(pat_ids, dtype=torch.long, device=device)
    return theta, dst, n_dst, pat_ids, list(pat_names)


def summary(data, phenotypes):
    """Print summary statistics of extracted transition data."""
    print(f"  Clones: {data['clone'].nunique()}")
    print(f"  Observations: {len(data)}")
    print(f"  Patients: {data['patient'].nunique()} — {sorted(data['patient'].unique())}")
    print(f"  Phenotypes ({len(phenotypes)}): {phenotypes}")
    print(f"  Cells — src: {data['n_src'].sum()}, dst: {data['n_dst'].sum()}")
    if "tp_src" in data.columns:
        pairs = data[data["temporal"]].groupby(["tp_src", "tp_dst"]).size()
        if len(pairs) > 0:
            print(f"  Timepoint pairs:")
            for (t1, t2), n in pairs.items():
                print(f"    {t1} → {t2}: {n} observations")
