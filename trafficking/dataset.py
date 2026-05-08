import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path


TISSUE_MAP = {"PBMC": "Plasma", "Blood": "Plasma", "Tumor": "TP"}
TISSUE_ORDER = ["Plasma", "CSF", "TP"]
TISSUE_LABELS = {"Plasma": "Blood", "CSF": "CSF", "TP": "Tumor"}


def _standardize_tissue(adata):
    adata = adata.copy()
    adata.obs["tissue"] = adata.obs["tissue"].replace(TISSUE_MAP)
    return adata


class GBMDataset:
    """Unified interface to T cell, myeloid, and olink data for a GBM cohort."""

    def __init__(self, tdata=None, mdata=None, odata=None,
                 phenotype_key="phenotype", clone_key="trb",
                 tissue_key="tissue", patient_key="patient",
                 time_key="timepoint", myeloid_phenotype_key=None):
        self._keys = dict(phenotype=phenotype_key, clone=clone_key,
                          tissue=tissue_key, patient=patient_key,
                          time=time_key)

        self.tdata = _standardize_tissue(tdata) if tdata is not None else None
        self.mdata = _standardize_tissue(mdata) if mdata is not None else None
        if mdata is not None and myeloid_phenotype_key and myeloid_phenotype_key != phenotype_key:
            self.mdata.obs[phenotype_key] = self.mdata.obs[myeloid_phenotype_key]
        self.odata = odata

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_paths(cls, tdata_path=None, mdata_path=None, odata_path=None, **kwargs):
        import scanpy as sc
        tdata = sc.read_h5ad(tdata_path) if tdata_path else None
        mdata = sc.read_h5ad(mdata_path) if mdata_path else None
        odata = sc.read_h5ad(odata_path) if odata_path else None
        return cls(tdata=tdata, mdata=mdata, odata=odata, **kwargs)

    @staticmethod
    def build_olink_adata(batch1_path, batch2_path, drop_controls=True):
        df = pd.read_csv(batch1_path, sep="\t")
        df2 = pd.read_csv(batch2_path, sep="\t")

        for d in (df, df2):
            d.index = d["Panel"]
            del d["Panel"]
            d.columns = d.loc["Assay"]
        df = df.loc[:, ~df.columns.duplicated()]
        df2 = df2.loc[:, ~df2.columns.duplicated()]
        df = df.drop(df.index[:6])
        df2 = df2.drop(df2.index[:6])
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        df2 = df2.apply(pd.to_numeric, errors="coerce").fillna(0)

        common = df.index.intersection(df2.index)
        if len(common) == 0:
            raise ValueError("No shared sample between batches for normalization.")
        ref = common[0]
        scaling = df.loc[df.index == ref].iloc[0] / df2.loc[df2.index == ref].iloc[0].replace(0, np.nan)
        scaling = scaling.fillna(1)
        df2_norm = df2.multiply(scaling, axis=1)

        adata1 = ad.AnnData(X=df.values, obs=pd.DataFrame(index=df.index),
                            var=pd.DataFrame(index=df.columns))
        adata2 = ad.AnnData(X=df2_norm.values, obs=pd.DataFrame(index=df2_norm.index),
                            var=pd.DataFrame(index=df2_norm.columns))
        adata2.var_names_make_unique()
        adata = ad.concat([adata1, adata2], keys=["batch1", "batch2"],
                          label="batch", merge="same")

        pts, tps = [], []
        for x in adata.obs.index:
            x = x.replace(" ", ".")
            if ".CF." not in x:
                parts = x.split(".")
                pts.append(parts[1] if len(parts) > 1 else "unknown")
                tps.append(parts[2] if len(parts) > 2 else "unknown")
            else:
                pts.append("Control")
                tps.append("Control")
        adata.obs["patient"] = pts
        adata.obs["timepoint"] = tps

        if drop_controls:
            adata = adata[~adata.obs["patient"].str.contains("Control")].copy()
        return adata

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tissues(self):
        vals = set()
        for ad_ in (self.tdata, self.mdata):
            if ad_ is not None:
                vals.update(ad_.obs[self._keys["tissue"]].unique())
        return sorted(vals, key=lambda x: TISSUE_ORDER.index(x) if x in TISSUE_ORDER else 99)

    @property
    def timepoints(self):
        vals = set()
        for ad_ in (self.tdata, self.mdata, self.odata):
            if ad_ is not None:
                vals.update(ad_.obs[self._keys["time"]].unique())
        return sorted(vals)

    @property
    def patients(self):
        vals = set()
        for ad_ in (self.tdata, self.mdata, self.odata):
            if ad_ is not None:
                vals.update(ad_.obs[self._keys["patient"]].unique())
        return sorted(vals)

    def phenotypes(self, modality="tcell"):
        ad_ = self._get_adata(modality)
        return sorted(ad_.obs[self._keys["phenotype"]].unique()) if ad_ is not None else []

    @property
    def proteins(self):
        return list(self.odata.var_names) if self.odata is not None else []

    # ------------------------------------------------------------------
    # Core queries
    # ------------------------------------------------------------------

    def _get_adata(self, modality):
        return {"tcell": self.tdata, "myeloid": self.mdata, "olink": self.odata}.get(modality)

    def _filter(self, adata, tissue=None, timepoint=None, patient=None):
        mask = np.ones(len(adata), dtype=bool)
        if tissue is not None:
            mask &= (adata.obs[self._keys["tissue"]] == tissue).values
        if timepoint is not None:
            mask &= (adata.obs[self._keys["time"]] == timepoint).values
        if patient is not None:
            mask &= (adata.obs[self._keys["patient"]] == patient).values
        return adata[mask]

    def get_cells(self, modality="tcell", tissue=None, timepoint=None, patient=None):
        ad_ = self._get_adata(modality)
        if ad_ is None:
            raise ValueError(f"No {modality} data loaded")
        return self._filter(ad_, tissue, timepoint, patient)

    def get_composition(self, modality="tcell", tissue=None, timepoint=None,
                        patient=None, normalize=True):
        sub = self.get_cells(modality, tissue, timepoint, patient)
        counts = sub.obs[self._keys["phenotype"]].value_counts()
        if normalize and len(counts) > 0:
            counts = counts / counts.sum()
        return counts.sort_index()

    def get_composition_matrix(self, modality="tcell", tissue=None, normalize=True):
        ad_ = self._get_adata(modality)
        if ad_ is None:
            return pd.DataFrame()
        sub = self._filter(ad_, tissue=tissue)
        pk, tk, patk = self._keys["phenotype"], self._keys["time"], self._keys["patient"]
        groups = sub.obs.groupby([patk, tk], observed=True)[pk].value_counts()
        if normalize:
            totals = sub.obs.groupby([patk, tk], observed=True)[pk].count()
            groups = groups / totals
        return groups.unstack(fill_value=0)

    # ------------------------------------------------------------------
    # Clone queries
    # ------------------------------------------------------------------

    def get_clones(self, tissue=None, timepoint=None, patient=None, lineage=None):
        if self.tdata is None:
            raise ValueError("No T cell data loaded")
        sub = self._filter(self.tdata, tissue, timepoint, patient)
        obs = sub.obs.dropna(subset=[self._keys["clone"]]).copy()
        if lineage is not None:
            lin_col = obs[self._keys["phenotype"]].apply(
                lambda x: "CD8" if "CD8" in str(x) else "CD4")
            obs = obs[lin_col == lineage]
        return obs

    def get_clone_set(self, tissue=None, timepoint=None, patient=None, lineage=None):
        obs = self.get_clones(tissue, timepoint, patient, lineage)
        return set(obs[self._keys["clone"]])

    def get_shared_clones(self, t1, t2, timepoint=None, patient=None, lineage=None):
        c1 = self.get_clone_set(t1, timepoint, patient, lineage)
        c2 = self.get_clone_set(t2, timepoint, patient, lineage)
        return c1 & c2

    def clone_phenotype_counts(self, tissue=None, timepoint=None, patient=None,
                               lineage=None):
        obs = self.get_clones(tissue, timepoint, patient, lineage)
        ck, pk = self._keys["clone"], self._keys["phenotype"]
        return obs.groupby(ck, observed=True)[pk].value_counts().unstack(fill_value=0)

    # ------------------------------------------------------------------
    # Olink queries
    # ------------------------------------------------------------------

    def get_olink(self, patient=None, timepoint=None):
        if self.odata is None:
            raise ValueError("No olink data loaded")
        sub = self._filter(self.odata, patient=patient, timepoint=timepoint)
        return pd.DataFrame(sub.X, index=sub.obs.index, columns=self.odata.var_names)

    def get_olink_trajectory(self, patient, proteins=None):
        if self.odata is None:
            raise ValueError("No olink data loaded")
        sub = self._filter(self.odata, patient=patient)
        df = pd.DataFrame(sub.X, index=sub.obs[self._keys["time"]].values,
                          columns=self.odata.var_names)
        df = df.groupby(level=0).mean().sort_index()
        if proteins is not None:
            df = df[[p for p in proteins if p in df.columns]]
        return df

    def get_olink_matrix(self, proteins=None):
        if self.odata is None:
            raise ValueError("No olink data loaded")
        df = pd.DataFrame(self.odata.X, columns=self.odata.var_names,
                          index=self.odata.obs.index)
        df["patient"] = self.odata.obs[self._keys["patient"]].values
        df["timepoint"] = self.odata.obs[self._keys["time"]].values
        df = df.groupby(["patient", "timepoint"]).mean()
        if proteins is not None:
            df = df[[p for p in proteins if p in df.columns]]
        return df

    # ------------------------------------------------------------------
    # Cross-modality: state vector for CTMC
    # ------------------------------------------------------------------

    def get_state_vector(self, modality="tcell", timepoint=None, patient=None,
                         lineage=None, normalize=True):
        records = {}
        for tis in self.tissues:
            comp = self.get_composition(modality, tissue=tis, timepoint=timepoint,
                                        patient=patient, normalize=normalize)
            for pheno, val in comp.items():
                records[(tis, pheno)] = val
        s = pd.Series(records)
        s.index = pd.MultiIndex.from_tuples(s.index, names=["tissue", "phenotype"])
        return s

    def get_state_matrix(self, modality="tcell", patient=None, lineage=None,
                         normalize=True):
        rows = {}
        for tp in self.timepoints:
            rows[tp] = self.get_state_vector(modality, timepoint=tp, patient=patient,
                                             lineage=lineage, normalize=normalize)
        return pd.DataFrame(rows).T.fillna(0)

    def get_covariates(self, timepoint, patient=None, myeloid_tissue=None):
        cov = {}
        if self.mdata is not None:
            comp = self.get_composition("myeloid", tissue=myeloid_tissue,
                                        timepoint=timepoint, patient=patient)
            for k, v in comp.items():
                cov[f"myeloid_{k}"] = v
        if self.odata is not None:
            olink = self.get_olink(patient=patient, timepoint=timepoint)
            if len(olink) > 0:
                for prot in olink.columns:
                    cov[f"olink_{prot}"] = olink[prot].mean()
        return pd.Series(cov)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self):
        lines = ["GBMDataset Summary", "=" * 40]
        lines.append(f"Patients: {len(self.patients)} — {self.patients}")
        lines.append(f"Timepoints: {len(self.timepoints)} — {self.timepoints}")
        lines.append(f"Tissues: {self.tissues}")
        if self.tdata is not None:
            lines.append(f"T cells: {len(self.tdata):,} cells, "
                         f"{len(self.phenotypes('tcell'))} phenotypes, "
                         f"{self.tdata.obs[self._keys['clone']].dropna().nunique():,} clones")
        if self.mdata is not None:
            lines.append(f"Myeloid: {len(self.mdata):,} cells, "
                         f"{len(self.phenotypes('myeloid'))} phenotypes")
        if self.odata is not None:
            lines.append(f"Olink: {len(self.odata)} samples, {len(self.proteins)} proteins")
        out = "\n".join(lines)
        print(out)
        return out

    def __repr__(self):
        parts = []
        if self.tdata is not None:
            parts.append(f"T={len(self.tdata):,}")
        if self.mdata is not None:
            parts.append(f"M={len(self.mdata):,}")
        if self.odata is not None:
            parts.append(f"O={len(self.odata)}")
        return f"GBMDataset({', '.join(parts)})"