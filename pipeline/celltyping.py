import collections
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import tqdm
from scipy import sparse
from scipy.stats import entropy

from modules.clone_helpers import infer_lineage_from_phenotype

def load_trb(sample, gex_h5, vdj_file, clonotype_file):
    adata = sc.read_10x_h5(gex_h5)
    contigs = pd.read_csv(vdj_file)
    barcodes = dict()
    for barcode, clonotype in zip(contigs["barcode"],contigs["raw_clonotype_id"]):
        barcodes[barcode] = clonotype    
    assert os.path.exists(clonotype_file)
    clonotypes = pd.read_csv(clonotype_file)
    print(clonotypes.columns.tolist())
    seqs = dict()
    inkt = dict()
    mait = dict()
    for clonotype, seq in zip(clonotypes["clonotype_id"],clonotypes["cdr3s_aa"]):
        seqs[clonotype] = seq
    for clonotype, seq in zip(clonotypes["clonotype_id"],clonotypes["inkt_evidence"]):
        inkt[clonotype] = seq
    for clonotype, seq in zip(clonotypes["clonotype_id"],clonotypes["mait_evidence"]):
        mait[clonotype] = seq
    trbs = []
    maits = []
    inkts = []
    for barcode in adata.obs.index:
        if barcode in barcodes:
            maits.append(mait[barcodes[barcode]])
            inkts.append(inkt[barcodes[barcode]])
            seq = seqs[barcodes[barcode]]
            if "TRB:" not in seq:
                trbs.append("None")
            else:
                flag = False
                chains = seq.split(";")
                for chain in chains:
                    if "TRB:" in chain:
                        trbs.append(chain.replace("TRB:",""))
                        flag = True
                        break
                if not flag: print("BAD")
        else:
            trbs.append("None")
            maits.append("None")
            inkts.append("None")
    adata.obs['mait'] = maits
    adata.obs['inkt'] = inkts
    
    tras = []
    for barcode in adata.obs.index:
        if barcode in barcodes:
            seq = seqs[barcodes[barcode]]
            if "TRA:" not in seq:
                tras.append("None")
            else:
                flag = False
                chains = seq.split(";")
                for chain in chains:
                    if "TRA:" in chain:
                        tras.append(chain.replace("TRA:",""))
                        flag = True
                        break
                if not flag: print("BAD")
        else:
            tras.append("None")
    
    adata.obs['trb'] = trbs
    adata.obs['tra'] = tras
    adata.var_names_make_unique()
    print(sample,"complete",len(set(adata.obs["trb"])))
    return adata.copy()

def normalized_exponential_vector(values, temperature=0.01):
    exps = np.exp(values / temperature)
    return exps / np.sum(exps)

def _score_cell(adata, idx, genes):
    expr = []
    for gene in genes:
        if gene in adata.var_names:
            val = adata.X[idx, adata.var_names.get_loc(gene)]
            if sparse.issparse(adata.X):
                val = val.toarray().flatten()[0] if hasattr(val, 'toarray') else val
            expr.append(val)
    return np.mean(expr) if expr else 0

def classify_recursive(adata, markers, temperature=0.01, key="phenotype"):
    level1_markers = {ct: info["markers"] for ct, info in markers.items()}
    for ct, genes in level1_markers.items():
        sc.tl.score_genes(adata, score_name=f"{ct}_L1_SCORE", gene_list=genes)

    level1_mat = adata.obs[[f"{ct}_L1_SCORE" for ct in markers]].to_numpy()
    level1_types = []
    for x in level1_mat:
        probs = normalized_exponential_vector(x, temperature=temperature)
        level1_types.append(list(markers.keys())[np.argmax(probs)])
    adata.obs["phenotype_level1"] = level1_types

    n_cells = adata.n_obs
    final_labels = [""] * n_cells

    def _classify_level(cell_indices, subtypes_dict, prefix, depth):
        if not cell_indices:
            return
        subtype_names = list(subtypes_dict.keys())
        is_leaf = {st: isinstance(subtypes_dict[st], list) for st in subtype_names}
        for i in cell_indices:
            scores = []
            for st in subtype_names:
                genes = subtypes_dict[st] if is_leaf[st] else subtypes_dict[st]["markers"]
                scores.append(_score_cell(adata, i, genes))
            probs = normalized_exponential_vector(np.array(scores), temperature=temperature)
            best = subtype_names[np.argmax(probs)]
            if is_leaf[best]:
                final_labels[i] = f"{prefix}_{best}"
            else:
                _classify_level([i], subtypes_dict[best]["subtypes"], f"{prefix}_{best}", depth + 1)

    l1_groups = collections.defaultdict(list)
    for i, l1 in enumerate(level1_types):
        l1_groups[l1].append(i)
    for l1_type, indices in l1_groups.items():
        if "subtypes" in markers[l1_type]:
            _classify_level(indices, markers[l1_type]["subtypes"], l1_type, 2)
        else:
            for i in indices:
                final_labels[i] = l1_type
    adata.obs[key] = final_labels
    return adata

def apply_temra_gate(adata, key="phenotype"):
    cx3cr1 = adata[:, "CX3CR1"].X.toarray().flatten() if sparse.issparse(adata.X) else adata[:, "CX3CR1"].X.flatten()
    s1pr5 = adata[:, "S1PR5"].X.toarray().flatten() if sparse.issparse(adata.X) else adata[:, "S1PR5"].X.flatten()
    klf2 = adata[:, "KLF2"].X.toarray().flatten() if sparse.issparse(adata.X) else adata[:, "KLF2"].X.flatten()
    mask = adata.obs[key] == "CD8_Activated_TEXeff"
    temra_gate = mask & (cx3cr1 > 0) & ((s1pr5 > 0) | (klf2 > 1.0))
    n_temra = temra_gate.sum()
    adata.obs.loc[temra_gate, key] = "CD8_Activated_TEMRA"
    return n_temra

def correct_lineage_switching(adata, markers, clone_key="trb", key="phenotype", temperature=0.01):
    tcr = adata.obs[adata.obs[clone_key].notna()].copy()
    tcr["_lineage"] = tcr[key].apply(infer_lineage_from_phenotype)

    clone_lineage = tcr.groupby(clone_key)["_lineage"].nunique()
    switching = clone_lineage[clone_lineage > 1].index
    sw = tcr[tcr[clone_key].isin(switching)]

    if len(switching) == 0:
        print("No switching clones found.")
        return 0

    clone_majority = sw.groupby(clone_key)["_lineage"].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else None).dropna()

    # Find minority cells
    fix_map = {}
    for clone, majority_lin in clone_majority.items():
        clone_cells = sw[sw[clone_key] == clone]
        minority = clone_cells[clone_cells["_lineage"] != majority_lin]
        for bc in minority.index:
            fix_map[bc] = majority_lin

    if not fix_map:
        print("No cells to reclassify.")
        return 0

    fix_barcodes = list(fix_map.keys())
    print(f"Cells to reclassify: {len(fix_barcodes)}")
    print(f"Direction: {pd.Series(fix_map).value_counts().to_dict()}")

    # Reclassify each cell within its corrected lineage
    cd4_subtypes = {st: genes for st, genes in markers["CD4"]["subtypes"].items()}
    cd8_subtypes = markers["CD8"]["subtypes"]
    act_markers = cd8_subtypes["Activated"]["markers"]
    qui_markers = cd8_subtypes["Quiescent"]["markers"]
    cd8_act = cd8_subtypes["Activated"]["subtypes"]
    cd8_qui = cd8_subtypes["Quiescent"]["subtypes"]

    new_labels = {}
    for bc in fix_barcodes:
        idx = adata.obs.index.get_loc(bc)
        corrected_lin = fix_map[bc]

        if corrected_lin == "CD8":
            act_score = np.mean([_score_cell(adata, idx, [g]) for g in act_markers])
            qui_score = np.mean([_score_cell(adata, idx, [g]) for g in qui_markers])
            if act_score > qui_score:
                scores = {st: _score_cell(adata, idx, genes) for st, genes in cd8_act.items()}
                best = max(scores, key=scores.get)
                new_labels[bc] = f"CD8_Activated_{best}"
            else:
                scores = {st: _score_cell(adata, idx, genes) for st, genes in cd8_qui.items()}
                best = max(scores, key=scores.get)
                new_labels[bc] = f"CD8_Quiescent_{best}"
        else:
            scores = {st: _score_cell(adata, idx, genes) for st, genes in cd4_subtypes.items()}
            best = max(scores, key=scores.get)
            new_labels[bc] = f"CD4_{best}"

    for bc, label in new_labels.items():
        adata.obs.loc[bc, key] = label

    return len(fix_barcodes)

def load_markers(beltra_path="41586_2025_9989_MOESM10_ESM.xlsx"):
    df_sig = pd.read_excel(beltra_path, header=None)
    tex = df_sig.iloc[3:, :2].dropna()
    tex.columns = ["gene", "signature"]
    trm = df_sig.iloc[3:, 3:5].dropna()
    trm.columns = ["gene", "signature"]
    gene_sigs = {}
    for sig in tex["signature"].unique():
        gene_sigs[sig] = tex[tex["signature"] == sig]["gene"].tolist()
    for sig in trm["signature"].unique():
        gene_sigs[sig] = trm[trm["signature"] == sig]["gene"].tolist()
    to_human = lambda genes: [g.upper() for g in genes]

    return {
        "CD8": {
            "markers": ["CD8A", "CD8B"],
            "subtypes": {
                "Activated": {
                    "markers": ["NKG7", "GZMB", "PRF1", "GNLY", "CST7"],
                    "subtypes": {
                        "TEXeff": to_human(gene_sigs["TEX eff-like"]),
                        "TEXterm": to_human(gene_sigs["TEXterm"]),
                        "TRM": to_human(gene_sigs["TRM UP"]),
                    }
                },
                "Quiescent": {
                    "markers": ["TCF7", "SELL", "IL7R", "LEF1", "CCR7"],
                    "subtypes": {
                        "Naive": ["CCR7", "SELL", "LEF1", "MAL", "BACH2"],
                        "TEXprog": to_human(gene_sigs["TEXprog"]),
                        "Memory": ["GZMK", "EOMES", "GZMA", "GZMB"],
                    }
                },
            }
        },
        "CD4": {
            "markers": ["CD4"],
            "subtypes": {
                "Naive_Memory": ["IL7R", "TCF7", "MAL", "LEF1"],
                "Exhausted": ["PDCD1", "LAG3", "TIGIT", "TOX", "CXCL13", "CTLA4"],
                "Treg": ["FOXP3", "IL32", "CTLA4", "IKZF2", "IL2RA", "ICOS",
                         "ARID5B", "IL2RG", "TIGIT", "TNFRSF1B"],
                "Th1_polarized": ["TBX21", "STAT1", "CXCR3", "IFNG", "IL12RB2"],
                "Th2_polarized": ["GATA3", "STAT6", "IL4", "IL4R"],
            }
        },
    }

def phenotype_tcells(adata, beltra_path="41586_2025_9989_MOESM10_ESM.xlsx",
                     clone_key="trb", key="phenotype", temperature=0.01):
    markers = load_markers(beltra_path)

    # Step 1: Hierarchical classifier
    print("=" * 60)
    print("STEP 1: Hierarchical classifier")
    print("=" * 60)
    classify_recursive(adata, markers, temperature=temperature, key=key)
    print(adata.obs[key].value_counts())

    # Step 2: Post-hoc TEMRA gate
    print("\n" + "=" * 60)
    print("STEP 2: Post-hoc TEMRA gate (CX3CR1+ AND S1PR5/KLF2)")
    print("=" * 60)
    n_temra = apply_temra_gate(adata, key=key)
    print(f"TEMRA cells gated: {n_temra}")
    print(adata.obs[key].value_counts())

    # Step 3: Lineage switching correction
    print("\n" + "=" * 60)
    print("STEP 3: TCR lineage switching correction")
    print("=" * 60)
    n_fixed = correct_lineage_switching(adata, markers, clone_key=clone_key,
                                         key=key, temperature=temperature)
    # Re-apply TEMRA gate on reclassified cells
    if n_fixed > 0:
        n_temra2 = apply_temra_gate(adata, key=key)
        print(f"Post-correction TEMRA gate: {n_temra2} additional")
    print(f"\nFinal phenotypes:")
    print(adata.obs[key].value_counts())

    # Step 4: Verify
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    _verify(adata, clone_key=clone_key, key=key)

    return adata

def _verify(adata, clone_key="trb", key="phenotype"):
    # Remaining switching clones
    tcr = adata.obs[adata.obs[clone_key].notna()].copy()
    tcr["_lin"] = tcr[key].apply(infer_lineage_from_phenotype)
    cl = tcr.groupby(clone_key)["_lin"].nunique()
    n_sw = (cl > 1).sum()
    print(f"Remaining switching clones: {n_sw}")

    # Tissue distribution
    print("\nTissue distribution:")
    ct = pd.crosstab(adata.obs[key], adata.obs["tissue"], normalize="index")
    print(ct.round(2))

    # Key markers per phenotype
    check = ["CX3CR1", "S1PR5", "KLF2", "CD69", "PDCD1", "TOX", "TCF7", "FOXP3"]
    check = [g for g in check if g in adata.var_names]
    print(f"\nKey marker means:")
    rows = []
    for pheno in sorted(adata.obs[key].unique()):
        sub = adata[adata.obs[key] == pheno]
        row = {"phenotype": pheno, "n": sub.n_obs}
        for g in check:
            v = sub[:, g].X.toarray().mean() if sparse.issparse(sub.X) else sub[:, g].X.mean()
            row[g] = round(v, 2)
        rows.append(row)
    print(pd.DataFrame(rows).set_index("phenotype").to_string())

    # TEMRA sanity
    temra = adata[adata.obs[key] == "CD8_Activated_TEMRA"]
    if temra.n_obs > 0:
        pbmc_frac = (temra.obs["tissue"].astype(str) == "PBMC").mean()
        print(f"\nTEMRA: {temra.n_obs} cells, {pbmc_frac:.0%} PBMC")

def annotate(adata):
    adata.obs["patient"] = [x.split("-")[1] for x in adata.obs["sample"]]
    adata.obs["timepoint"] = [x.split("-")[2] for x in adata.obs["sample"]]
    adata.obs["tissue"] = [x.split("-")[-1] for x in adata.obs["sample"]]

    tp_mapper = {
        'S3': '3',
        'S4': '4', 
        'S6': '6',
        'S5': '5',
        'S2': '2',
        'S1': '1',
        '001': '1'
    }

    pat = []
    for x in adata.obs["patient"]:
        if x == "GBM":
            pat.append("MSK1")
        else:
            pat.append(x)
    adata.obs["patient"] = pat

    adata.obs["timepoint"] = [tp_mapper[x] for x in adata.obs["timepoint"]]
    stats = set(adata.obs["patient"]), set(adata.obs["timepoint"]), set(adata.obs["tissue"])
    print(stats)
    return adata.copy()

def gene_entropy(adata, key_added="entropy"):
    X = adata.layers["counts"].todense()
    X = np.array(X.T)
    gene_to_row = list(zip(adata.var.index.tolist(), X))
    entropies = []
    for _, exp in tqdm.tqdm(gene_to_row):
        counts = np.unique(exp, return_counts = True)
        entropies.append(entropy(counts[1][1:]))
    adata.var[key_added] = entropies

def remove_meaningless_genes(adata, include_mt=True, include_rp=True, include_mtrn=True, include_hsp=True, include_tcr=False):
    genes = [x for x in adata.var.index.tolist() if "RIK" not in x.upper()]
    genes = [x for x in genes if "GM" not in x]
    genes = [x for x in genes if "-" not in x or "HLA" in x]
    genes = [x for x in genes if "." not in x or "HLA" in x]
    genes = [x for x in genes if "LINC" not in x.upper()]
    if include_mtrn:
        genes = [x for x in genes if "MTRN" not in x]
    if include_hsp:
        genes = [x for x in genes if "HSP" not in x]
    if include_mt:
        genes = [x for x in genes if "MT-" not in x.upper()]
    if include_rp:
        genes = [x for x in genes if "RP" not in x.upper()]
    if include_tcr:
        genes = [x for x in genes if "TRAV" not in x]
        genes = [x for x in genes if "TRAJ" not in x]
        genes = [x for x in genes if "TRAD" not in x]

        genes = [x for x in genes if "TRBV" not in x]
        genes = [x for x in genes if "TRBJ" not in x]
        genes = [x for x in genes if "TRBD" not in x]

        genes = [x for x in genes if "TRGV" not in x]
        genes = [x for x in genes if "TRGJ" not in x]
        genes = [x for x in genes if "TRGD" not in x]

        genes = [x for x in genes if "TRDV" not in x]
        genes = [x for x in genes if "TRDJ" not in x]
        genes = [x for x in genes if "TRDD" not in x]
    print(len(genes),len(adata.var.index.tolist()))
    adata = adata[:,genes]
    return adata.copy()

def run_harmony_workflow(adata,batch_key):
    sc.tl.pca(adata)
    sce.pp.harmony_integrate(adata, batch_key)
    sc.pp.neighbors(adata,use_rep="X_pca_harmony")
    sc.tl.umap(adata)
    return adata.copy()
