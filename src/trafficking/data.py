# Save as: src/trafficking/data.py
import numpy as np
import pandas as pd
import scanpy as sc
from dataclasses import dataclass
from typing import Optional

@dataclass
class TraffickingData:
    tcell_counts: np.ndarray
    myeloid_counts: np.ndarray
    obs_mask: np.ndarray
    clone_sharing: Optional[np.ndarray] = None
    patients: np.ndarray = None
    timepoints: np.ndarray = None
    compartments: np.ndarray = None
    tcell_phenotypes: np.ndarray = None
    myeloid_phenotypes: np.ndarray = None


def load_trafficking_data(
    tcell_path: str, 
    myeloid_path: str,
    compartment_mapping: dict = None
) -> TraffickingData:
    if compartment_mapping is None:
        compartment_mapping = {'Plasma': 'Blood', 'TP': 'Tumor', 'CSF': 'CSF'}
    
    adata_t = sc.read_h5ad(tcell_path)
    adata_m = sc.read_h5ad(myeloid_path)
    
    adata_t.obs['tissue'] = adata_t.obs['tissue'].map(compartment_mapping)
    adata_m.obs['tissue'] = adata_m.obs['tissue'].map(compartment_mapping)
    
    compartments = np.array(['Blood', 'CSF', 'Tumor'])
    patients = np.sort(adata_t.obs['patient'].unique())
    timepoints = np.sort(adata_t.obs['timepoint'].unique())
    tcell_pheno = np.sort(adata_t.obs['phenotype'].unique())
    myeloid_pheno = np.sort(adata_m.obs['phenotype'].unique())
    
    n_p, n_t, n_c = len(patients), len(timepoints), len(compartments)
    n_kt, n_km = len(tcell_pheno), len(myeloid_pheno)
    
    p_map = {p: i for i, p in enumerate(patients)}
    t_map = {t: i for i, t in enumerate(timepoints)}
    c_map = {c: i for i, c in enumerate(compartments)}
    kt_map = {k: i for i, k in enumerate(tcell_pheno)}
    km_map = {k: i for i, k in enumerate(myeloid_pheno)}
    
    tcell_counts = np.zeros((n_p, n_t, n_c, n_kt), dtype=np.int32)
    myeloid_counts = np.zeros((n_p, n_t, n_c, n_km), dtype=np.int32)
    obs_mask = np.zeros((n_p, n_t, n_c), dtype=bool)
    
    for adata, counts, pheno_map in [
        (adata_t, tcell_counts, kt_map),
        (adata_m, myeloid_counts, km_map)
    ]:
        grouped = adata.obs.groupby(['patient', 'timepoint', 'tissue', 'phenotype']).size()
        for (pat, tp, tis, phe), count in grouped.items():
            if pat in p_map and tp in t_map and tis in c_map and phe in pheno_map:
                counts[p_map[pat], t_map[tp], c_map[tis], pheno_map[phe]] = count
                obs_mask[p_map[pat], t_map[tp], c_map[tis]] = True
    
    clone_sharing = compute_clone_sharing(adata_t, p_map, t_map, c_map, kt_map)
    
    return TraffickingData(
        tcell_counts=tcell_counts,
        myeloid_counts=myeloid_counts,
        obs_mask=obs_mask,
        clone_sharing=clone_sharing,
        patients=patients,
        timepoints=timepoints,
        compartments=compartments,
        tcell_phenotypes=tcell_pheno,
        myeloid_phenotypes=myeloid_pheno
    )


def compute_clone_sharing(adata, p_map, t_map, c_map, kt_map):
    n_p, n_t, n_c, n_k = len(p_map), len(t_map), len(c_map), len(kt_map)
    sharing = np.zeros((n_p, n_t, n_c, n_k, n_c), dtype=np.float32)
    
    obs = adata.obs
    obs = obs[obs['trb_unique'].notna() & (obs['trb_unique'] != '')]
    
    for pat, pat_df in obs.groupby('patient'):
        if pat not in p_map:
            continue
        pi = p_map[pat]
        
        for tp, tp_df in pat_df.groupby('timepoint'):
            if tp not in t_map:
                continue
            ti = t_map[tp]
            
            compartment_clones = {}
            for tis, tis_df in tp_df.groupby('tissue'):
                if tis in c_map:
                    compartment_clones[c_map[tis]] = set(tis_df['trb_unique'].unique())
            
            for tis, tis_df in tp_df.groupby('tissue'):
                if tis not in c_map:
                    continue
                ci = c_map[tis]
                
                for phe, phe_df in tis_df.groupby('phenotype'):
                    if phe not in kt_map:
                        continue
                    ki = kt_map[phe]
                    
                    clones_here = set(phe_df['trb_unique'].unique())
                    if len(clones_here) == 0:
                        continue
                    
                    for cj, clones_there in compartment_clones.items():
                        shared = len(clones_here & clones_there)
                        sharing[pi, ti, ci, ki, cj] = shared / len(clones_here)
    
    return sharing