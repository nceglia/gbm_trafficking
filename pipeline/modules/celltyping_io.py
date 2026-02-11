"""Celltyping-specific IO helpers."""

from __future__ import annotations

import glob
import os

import scanpy as sc
import scirpy as ir


def _concat_samples(adatas, samples):
    if not adatas:
        raise ValueError("No .h5 files found for loading.")
    if len(adatas) == 1:
        adata = adatas[0].copy()
        adata.obs["sample"] = samples[0]
        return adata
    return adatas[0].concatenate(adatas[1:], batch_key="sample", batch_categories=samples)


def load_directory_scirpy(directory):
    path = os.path.join(directory, "*.h5")
    adatas = []
    samples = []
    for h5_path in glob.glob(path):
        vdj_file = h5_path.replace(".h5", ".csv")
        adata = sc.read_10x_h5(h5_path)
        vdj = ir.io.read_10x_vdj(vdj_file)
        ir.pp.merge_with_ir(adata, vdj)
        adata.var_names_make_unique()
        adatas.append(adata)
        sample = os.path.split(h5_path)[1].replace(".h5", "")
        samples.append(sample)
        print(sample, "Loaded.")
    return _concat_samples(adatas, samples)


def load_directory_manual(directory, load_trb_fn):
    path = os.path.join(directory, "*.h5")
    adatas = []
    samples = []
    for h5_path in glob.glob(path):
        print(h5_path)
        sample = os.path.split(h5_path)[1].replace(".h5", "")
        vdj_file = h5_path.replace(".h5", ".csv")
        clonotype_file = vdj_file.replace(".csv", "_clonotypes.csv")
        print(clonotype_file)
        adata = load_trb_fn(sample, h5_path, vdj_file, clonotype_file)
        adata.var_names_make_unique()
        adatas.append(adata)
        samples.append(sample)
        print(sample, "Loaded.")
    return _concat_samples(adatas, samples)

