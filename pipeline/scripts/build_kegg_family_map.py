"""Build the KEGG -> pathway family map for the GBM trafficking analysis.

EXCLUSION RATIONALE
-------------------
This is a glioblastoma (GBM) cohort. KEGG contains many disease-annotated
pathways that share genes with immune/stress programs and would otherwise
inflate the inflammation/stress families with redundant signal. However,
because GBM is a brain tumor, several KEGG categories that look like
"disease noise" in a generic context are actually biologically relevant:

  KEPT (relevant to GBM TME biology):
    - Neurodegeneration (Alzheimer, Parkinson, Huntington, ALS, Prion):
      tumor cells and infiltrating T cells in brain parenchyma share
      mitochondrial stress and ER stress responses with neurodegeneration.
    - Cancer pathways (Glioma, Pathways in cancer, etc.): direct relevance.
    - Viral pathways (EBV, CMV, HIV, hepatitis, etc.): bystander viral
      reactivation in TILs is a real source of T cell signal.
    - Atherosclerosis / lipid disease: captures macrophage-T cell
      interactions and oxidized lipid stress in the TME.
    - Bacterial infection pathways (Tuberculosis, Salmonella, etc.):
      duplicate the TLR/NF-kB inflammation signal through shared genes.

  EXCLUDED (genuinely off-target for T cells in any cohort):
    - Cardiac and skeletal muscle contraction pathways
    - Organ-specific secretion/absorption (digestive, renal)
    - Substance abuse / behavioral neuroscience
    - Sensory perception (olfactory, visual, taste)
    - Parasitic infections endemic to non-US populations

To regenerate the analysis WITHOUT exclusions (full KEGG coverage), set
APPLY_EXCLUSIONS = False at the top of this script and re-run. The
sidecar file pathway_families_kegg_excluded.tsv lists every excluded term
along with its exclusion reason for audit purposes.

Usage:
    conda run -n scrna python pipeline/scripts/build_kegg_family_map.py
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))
from modules.style import PATHWAY_FAMILY_ORDER, EXCLUDED_FAMILY

# ── Config (must match 07_pathway_ternary.py) ────────────────────────
APPLY_EXCLUSIONS = True  # set False to retain ALL KEGG terms (no exclusions)

LINEAGES = ("CD8", "CD4")
DIRECTED_TISSUE_PAIRS = (("PBMC", "TP"), ("PBMC", "CSF"), ("CSF", "TP"))
INPUT_DIR = REPO_ROOT / "results" / "clone_pseudobulk_pathways"
MODULES_DIR = REPO_ROOT / "pipeline" / "modules"
FDR_THRESHOLD = 0.25
NES_THRESHOLD = 1.5
LIBRARY_NAME = "KEGG_2021_Human"
LIBRARY_TAG = "kegg"

# ── Narrowly scoped exclusion list (first match wins) ───────────────
EXCLUSION_CATEGORIES = {
    "Cardiac / muscle physiology": [
        "cardiac muscle contraction", "vascular smooth muscle contraction",
        "arrhythmogenic right ventricular", "hypertrophic cardiomyopathy",
        "dilated cardiomyopathy",
    ],
    "Organ-specific secretion / absorption": [
        "gastric acid secretion", "salivary secretion", "pancreatic secretion",
        "bile secretion", "insulin secretion",
        "aldosterone-regulated sodium reabsorption",
        "vasopressin-regulated water reabsorption",
        "proximal tubule bicarbonate reclamation",
        "collecting duct acid secretion",
        "renin-angiotensin",
        "carbohydrate digestion and absorption",
        "fat digestion and absorption",
        "protein digestion and absorption",
        "vitamin digestion and absorption",
        "mineral absorption",
    ],
    "Substance abuse / behavioral": [
        "cocaine addiction", "amphetamine addiction", "morphine addiction",
        "nicotine addiction", "alcoholism",
    ],
    "Sensory perception": [
        "olfactory transduction", "phototransduction", "taste transduction",
        "inflammatory mediator regulation of trp",
    ],
    "Parasitic infection (off-target for GBM)": [
        "african trypanosomiasis", "leishmaniasis", "chagas disease",
        "amoebiasis", "schistosomiasis", "malaria", "toxoplasmosis",
    ],
}

# ── Keyword rules (case-insensitive substring, first match wins) ────
KEYWORD_RULES = [
    # ── Immune / inflammatory ────────────────────────────────────────
    ("NF-kB / Inflammation", [
        "nf-kappa b", "tnf signaling", "inflammatory", "complement",
        "coagulation", "allograft rejection", "graft-versus-host",
        "osteoclast differentiation", "atherosclerosis",
        "rheumatoid arthritis",
    ]),
    ("Type I/II interferon", [
        "type i interferon", "type ii interferon", "rig-i",
        "cytosolic dna", "toll-like", "nod-like receptor",
        "c-type lectin receptor",
    ]),
    ("Cytokine signaling", [
        "cytokine-cytokine", "jak-stat", "il-17", "chemokine signaling",
        "viral protein interaction with cytokine", "adipocytokine",
    ]),
    ("TCR / antigen receptor", [
        "t cell receptor", "b cell receptor", "antigen processing",
        "th1 and th2", "th17", "natural killer", "pd-l1", "pd-1",
        "immunodeficiency", "iga production", "phagosome",
        "asthma", "autoimmune thyroid", "type i diabetes",
        "systemic lupus", "viral myocarditis",
    ]),
    # ── Infection pathways (innate immune / NF-kB activation) ────────
    ("NF-kB / Inflammation", [
        "influenza", "hepatitis", "herpes", "measles",
        "epstein-barr", "human cytomegalovirus", "human immunodeficiency",
        "human t-cell leukemia", "kaposi", "coronavirus",
        "pertussis", "legionellosis", "tuberculosis", "salmonella",
        "shigellosis", "yersinia", "vibrio", "toxoplasmosis",
        "leishmaniasis", "malaria", "chagas", "amoebiasis",
        "helicobacter pylori", "escherichia coli",
        "african trypanosomiasis",
        "infection",
    ]),
    # ── Stress / cell death ──────────────────────────────────────────
    ("Stress response", [
        "hypoxia", "p53", "unfolded protein",
        "protein processing in endoplasmic", "oxidative stress",
        "hif-1", "mitophagy",
        "alzheimer", "parkinson", "huntington", "amyotrophic",
        "prion", "neurodegeneration",
        "non-alcoholic fatty liver",
    ]),
    ("Apoptosis / cell death", [
        "apoptosis", "necroptosis", "ferroptosis", "autophagy",
    ]),
    # ── Proliferation / genome integrity ─────────────────────────────
    ("Proliferation / cell cycle", [
        "cell cycle", "dna replication", "mitotic",
        "spliceosome", "proteasome", "mrna surveillance",
    ]),
    ("DNA damage / repair", [
        "repair", "homologous recombination", "mismatch",
        "non-homologous end-joining", "fanconi anemia",
    ]),
    # ── Metabolism ───────────────────────────────────────────────────
    ("Metabolism — oxidative", [
        "oxidative phosphorylation", "citrate cycle",
        "fatty acid degradation", "propanoate", "butanoate", "tca",
        "valine, leucine", "lysine degradation",
        "arginine and proline metabolism", "fatty acid elongation",
        "thermogenesis", "retrograde endocannabinoid",
        "cardiac muscle contraction",
        "diabetic cardiomyopathy", "arrhythmogenic",
    ]),
    ("Metabolism — biosynthetic", [
        "glycolysis", "fatty acid biosynthesis", "mtor",
        "pentose phosphate", "amino sugar", "biosynthesis of amino acids",
        "pyrimidine metabolism", "purine metabolism", "steroid biosynthesis",
        "arginine biosynthesis", "glycine, serine",
        "glycosphingolipid", "cholesterol metabolism",
        "carbon metabolism", "phosphatidylinositol signaling",
        "ppar signaling", "abc transporter", "protein export",
        "mineral absorption", "lysosome",
    ]),
    # ── Adhesion / migration ─────────────────────────────────────────
    ("Migration / adhesion", [
        "cell adhesion", "focal adhesion", "leukocyte transendothelial",
        "regulation of actin", "tight junction", "gap junction",
        "ecm-receptor", "adherens junction",
    ]),
    # ── Signaling ────────────────────────────────────────────────────
    ("Growth factor signaling", [
        "pi3k", "mapk signaling", "ras signaling", "rap1", "egfr", "erbb",
        "mtor signaling", "tgf-beta",
        "age-rage signaling", "foxo signaling", "camp signaling",
        "relaxin signaling", "prolactin signaling",
        "pathways in cancer", "cancer", "carcinoma", "melanoma", "glioma",
        "leukemia", "sarcoma",
        "carcinogenesis", "transcriptional misregulation",
        "adrenergic signaling",
    ]),
    ("Hormone response", [
        "estrogen", "insulin signaling", "thyroid hormone", "gnrh",
        "diabetes mellitus", "vasopressin",
        "aldosterone", "parathyroid", "growth hormone",
    ]),
    ("Tissue morphogenesis", [
        "hippo", "wnt", "notch", "hedgehog", "vegf", "axon guidance",
        "basal cell carcinoma",
    ]),
    # ── Catch-alls (last resort; live when APPLY_EXCLUSIONS=False) ───
    ("Proliferation / cell cycle", [
        "circadian rhythm",
    ]),
    ("Growth factor signaling", [
        "addiction", "alcoholism",
    ]),
]


def check_exclusion(term):
    """Return (exclusion_reason, matched_keyword) or (None, None)."""
    if not APPLY_EXCLUSIONS:
        return None, None
    t_lower = term.lower()
    for reason, keywords in EXCLUSION_CATEGORIES.items():
        for kw in keywords:
            if kw in t_lower:
                return reason, kw
    return None, None


def classify_term(term):
    """Return family name or None if no keyword rule matches."""
    t_lower = term.lower()
    for family, keywords in KEYWORD_RULES:
        for kw in keywords:
            if kw in t_lower:
                return family
    return None


def load_filtered_kegg_terms():
    """Load KEGG GSEA CSVs and return list of terms passing NES/FDR filters."""
    records = []
    prefix = LIBRARY_NAME + "__"
    for lineage in LINEAGES:
        for t1, t2 in DIRECTED_TISSUE_PAIRS:
            label = f"{lineage}_{t1}_vs_{t2}"
            path = INPUT_DIR / f"gsea_{LIBRARY_TAG}_{label}.csv"
            if not path.exists():
                print(f"  SKIP (not found): {path.name}")
                continue
            gsea = pd.read_csv(path)
            gsea["lineage"] = lineage
            gsea["pathway"] = gsea["Term"].str.replace(prefix, "", regex=False)
            records.append(gsea)

    if not records:
        print("ERROR: No KEGG GSEA CSVs found!")
        sys.exit(1)

    gsea_long = pd.concat(records, ignore_index=True)
    stats = gsea_long.groupby(["lineage", "pathway"]).agg(
        max_abs_nes=("NES", lambda x: x.abs().max()),
        min_fdr=("FDR q-val", "min"),
    ).reset_index()

    passing = stats[
        (stats["max_abs_nes"] >= NES_THRESHOLD)
        & (stats["min_fdr"] <= FDR_THRESHOLD)
    ]

    terms = sorted(passing["pathway"].unique())
    print(f"Total KEGG terms in filtered results: {len(terms)}")
    return terms


def main():
    terms = load_filtered_kegg_terms()

    matched = {}
    excluded = []  # (term, reason)
    unmapped = []

    for term in terms:
        reason, _ = check_exclusion(term)
        if reason is not None:
            excluded.append((term, reason))
            matched[term] = EXCLUDED_FAMILY
            continue
        family = classify_term(term)
        if family is not None:
            matched[term] = family
        else:
            unmapped.append(term)

    # Validate families
    valid_families = set(PATHWAY_FAMILY_ORDER) | {EXCLUDED_FAMILY}
    bad = {f for f in matched.values() if f not in valid_families}
    if bad:
        print(f"ERROR: Unknown families in mapping: {bad}")
        sys.exit(1)

    # ── Write main TSV ──────────────────────────────────────────────
    tsv_path = MODULES_DIR / "pathway_families_kegg.tsv"
    with open(tsv_path, "w") as f:
        f.write("term\tfamily\n")
        for term in sorted(matched.keys()):
            f.write(f"{term}\t{matched[term]}\n")

    # ── Write excluded sidecar TSV ──────────────────────────────────
    excluded_path = MODULES_DIR / "pathway_families_kegg_excluded.tsv"
    with open(excluded_path, "w") as f:
        f.write("term\texclusion_reason\n")
        for term, reason in sorted(excluded):
            f.write(f"{term}\t{reason}\n")

    # ── Write unmapped TSV ──────────────────────────────────────────
    unmapped_path = MODULES_DIR / "pathway_families_kegg_unmapped.tsv"
    with open(unmapped_path, "w") as f:
        f.write("term\tsuggested_family\treason\n")
        for term in sorted(unmapped):
            f.write(f"{term}\t\tno keyword match\n")

    # ── Exclusion summary ───────────────────────────────────────────
    n_classified = len([t for t in terms if t in matched and matched[t] != EXCLUDED_FAMILY])
    if APPLY_EXCLUSIONS and excluded:
        print(f"\n{'=' * 60}")
        print("KEGG exclusion summary")
        print(f"{'=' * 60}")
        reason_counter = Counter(reason for _, reason in excluded)
        reason_terms = {}
        for term, reason in excluded:
            reason_terms.setdefault(reason, []).append(term)
        print(f"  {'Reason':<45} {'Count':>5}   Examples")
        for reason in EXCLUSION_CATEGORIES:
            cnt = reason_counter.get(reason, 0)
            if cnt == 0:
                continue
            examples = ", ".join(reason_terms[reason][:3])
            if cnt > 3:
                examples += ", ..."
            print(f"  {reason:<45} {cnt:>5}   {examples}")
        print(f"  {'─' * 60}")
        print(f"  {'Total excluded':<45} {len(excluded):>5}")
        print(f"  {'Total KEGG terms processed':<45} {len(terms):>5}")
        print(f"  {'Total mapped to families':<45} {n_classified:>5}")
    elif not APPLY_EXCLUSIONS:
        print(f"\n  APPLY_EXCLUSIONS = False → all {len(terms)} terms classified, "
              f"no exclusions applied.")

    # ── Family distribution ─────────────────────────────────────────
    counts = Counter()
    for t in terms:
        if t in matched:
            counts[matched[t]] += 1

    print(f"\nFamily distribution ({len(counts)} families, {len(terms)} terms):")
    for fam in PATHWAY_FAMILY_ORDER:
        n = counts.get(fam, 0)
        if n > 0:
            print(f"  {fam}: {n}")
    n_excl = counts.get(EXCLUDED_FAMILY, 0)
    if n_excl > 0:
        print(f"  {EXCLUDED_FAMILY}: {n_excl}")

    # ── Unmapped ────────────────────────────────────────────────────
    if unmapped:
        print(f"\nUnmapped terms ({len(unmapped)}):")
        for t in sorted(unmapped):
            print(f"  {t}")
    else:
        print("\nAll filtered terms mapped!")

    print(f"\nWrote {len(matched)} terms to {tsv_path}")
    print(f"Wrote {len(excluded)} excluded terms to {excluded_path}")


if __name__ == "__main__":
    main()
