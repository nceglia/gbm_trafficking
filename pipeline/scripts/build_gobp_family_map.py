"""Build pathway_families_gobp.tsv via GO slim ancestor mapping + keyword fallback.

Strategy:
  1. For each filtered GO BP term, try goatools.mapslim to find the closest
     slim ancestor in goslim_generic.
  2. Terms with no slim ancestor get classified by keyword matching on the
     term name against a curated rule set derived from slim family names.
  3. Small families (< MIN_FAMILY_SIZE) are merged into the nearest large one.

Downloads go-basic.obo and goslim_generic.obo on first run.

Usage:
    conda run -n scrna python pipeline/scripts/build_gobp_family_map.py
"""
import os
import re
import ssl
import sys
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))
from modules.style import EXCLUDED_FAMILY

# ── Config ────────────────────────────────────────────────────────────
LINEAGES = ("CD8", "CD4")
DIRECTED_TISSUE_PAIRS = (("PBMC", "TP"), ("PBMC", "CSF"), ("CSF", "TP"))
INPUT_DIR = REPO_ROOT / "results" / "clone_pseudobulk_pathways"
MODULES_DIR = REPO_ROOT / "pipeline" / "modules"
OBO_DIR = REPO_ROOT / "pipeline" / "data" / "ontology"
FDR_THRESHOLD = 0.25
NES_THRESHOLD = 1.5
LIBRARY_NAME = "GO_Biological_Process_2023"
LIBRARY_TAG = "gobp"
MIN_DEPTH = 3  # drop GO terms with depth <= 2 (too broad)
MIN_FAMILY_SIZE = 5  # families below this get merged

GO_ID_RE = re.compile(r"\(GO:\d{7}\)")

OBO_URLS = {
    "go-basic.obo": [
        "https://release.geneontology.org/2024-06-17/ontology/go-basic.obo",
        "https://current.geneontology.org/ontology/go-basic.obo",
    ],
    "goslim_generic.obo": [
        "https://current.geneontology.org/ontology/subsets/goslim_generic.obo",
        "https://release.geneontology.org/2024-06-17/ontology/subsets/goslim_generic.obo",
    ],
}

# ── Keyword fallback rules (for terms with no slim ancestor) ─────────
# Ordered: first match wins. Keywords are case-insensitive substrings.
KEYWORD_RULES = [
    ("immune system process", [
        "immune", "t cell", "b cell", "lymphocyte", "leukocyte",
        "macrophage", "monocyte", "dendritic cell", "nk cell",
        "antigen", "mhc", "cytokine", "chemokine", "interferon",
        "interleukin", "complement", "toll-like", "nod-like",
        "inflamm", "phagocyt", "opsoniz",
    ]),
    ("defense response to other organism", [
        "defense response", "viral", "virus", "antibacterial",
        "antimicrobial", "pathogen", "innate immune",
    ]),
    ("programmed cell death", [
        "apoptot", "apoptosis", "cell death", "necropt", "pyroptosis",
    ]),
    ("cell cycle", [
        "cell cycle", "mitotic", "meiotic", "cell division",
        "chromosome segregation", "spindle", "cytokinesis",
        "dna replication", "s phase", "g1/s", "g2/m",
    ]),
    ("DNA repair", [
        "dna repair", "dna damage", "double-strand break",
        "nucleotide excision", "base excision", "mismatch repair",
    ]),
    ("DNA recombination", [
        "recombination", "somatic diversification",
    ]),
    ("signaling", [
        "signal transduction", "signaling pathway", "signaling",
        "mapk", "wnt", "notch", "hedgehog", "jak-stat",
        "nf-kappa", "ras protein", "gtpase", "kinase cascade",
        "receptor signaling", "g protein-coupled",
    ]),
    ("regulation of DNA-templated transcription", [
        "transcription", "gene expression", "mrna processing",
        "rna splicing", "mrna transport", "polyadenylation",
    ]),
    ("chromatin organization", [
        "chromatin", "histone", "nucleosome", "epigenet",
    ]),
    ("lipid metabolic process", [
        "lipid", "fatty acid", "sterol", "cholesterol",
        "sphingolipid", "phospholipid", "prostaglandin",
        "leukotriene", "eicosanoid",
    ]),
    ("carbohydrate metabolic process", [
        "carbohydrate", "glyco", "sugar", "glucose",
        "glycolysis", "gluconeogenesis",
    ]),
    ("amino acid metabolic process", [
        "amino acid", "peptide", "protein metabol",
        "proteolysis", "ubiquitin", "proteasom",
    ]),
    ("nucleobase-containing small molecule metabolic process", [
        "nucleotide", "nucleoside", "purine", "pyrimidine",
        "nad", "atp", "gtp", "camp", "cgmp",
    ]),
    ("generation of precursor metabolites and energy", [
        "oxidative phosphorylation", "electron transport",
        "respiratory chain", "atp synthesis", "energy",
        "mitochondrial", "metabolism", "metabolic",
        "oxidation-reduction", "redox", "reactive oxygen",
        "oxidative stress",
    ]),
    ("transmembrane transport", [
        "transport", "ion channel", "ion transport",
        "calcium", "potassium", "sodium", "proton",
        "transporter", "pump", "channel",
    ]),
    ("vesicle-mediated transport", [
        "vesicle", "endocytosis", "exocytosis", "secretion",
        "endosome", "golgi", "lysosome",
    ]),
    ("membrane organization", [
        "membrane", "organelle", "autophagy",
    ]),
    ("cell motility", [
        "migration", "motility", "locomotion", "chemotaxis",
        "cell movement", "taxis",
    ]),
    ("cell adhesion", [
        "adhesion", "integrin", "cadherin", "junction",
    ]),
    ("cytoskeleton organization", [
        "cytoskelet", "actin", "microtubule", "tubulin",
        "myosin", "intermediate filament",
    ]),
    ("anatomical structure development", [
        "development", "morphogenesis", "differentiation",
        "embryo", "angiogenesis", "vasculature",
        "organ", "tissue", "pattern",
    ]),
    ("cell differentiation", [
        "differentiat", "stem cell", "lineage", "fate",
    ]),
    ("protein folding", [
        "protein folding", "chaperone", "heat shock",
        "unfolded protein", "endoplasmic reticulum stress",
    ]),
    ("protein-containing complex assembly", [
        "complex assembly", "ribosom", "spliceosom",
        "proteasome assembly",
    ]),
    ("nervous system process", [
        "synap", "neuron", "neural", "axon", "dendrit",
    ]),
    # ── Broader catch-alls (must come AFTER specific rules) ──────────
    # "Cellular response to X" — mostly signaling/stress
    ("signaling", [
        "cellular response to", "response to growth factor",
        "response to cytokine", "response to peptide",
        "response to hormone", "response to epidermal",
        "response to fibroblast", "response to insulin",
        "response to nerve growth", "response to lipopolysaccharide",
        "response to molecule of bacterial",
        "response to organonitrogen", "response to nitrogen",
        "response to organic cyclic", "response to oxygen-containing",
        "response to stimulus", "response to drug",
        "response to wounding", "response to organic substance",
        "response to endogenous", "response to external",
        "response to biotic",
        "phosphorylation", "dephosphorylation",
    ]),
    ("generation of precursor metabolites and energy", [
        "response to hypoxia", "response to decreased oxygen",
        "response to reactive oxygen", "response to oxidative",
        "response to hydrogen peroxide",
        "response to starvation", "response to nutrient",
        "response to heat", "response to cold",
        "response to temperature",
        "response to metal", "response to cadmium",
        "response to arsenic", "response to copper",
        "response to salt", "response to osmotic",
        "response to corticosteroid", "response to glucocorticoid",
        "response to catecholamine",
        "cellular response to misfolded protein",
        "stress", "homeostasis",
    ]),
    ("membrane organization", [
        "autophagosome", "phagophore", "autophagy",
    ]),
    ("regulation of DNA-templated transcription", [
        "mrna destabilization", "mrna stabilization",
        "mrna decay", "mrna degradation",
        "rna processing", "rna modification", "rna editing",
        "base conversion", "polyadenyl",
    ]),
    ("cell differentiation", [
        "cell fate", "commitment", "specification",
        "astrocyte", "osteoblast", "osteoclast",
        "adipocyte", "myoblast", "erythrocyte",
        "megakaryocyte", "myeloid", "granulocyte",
    ]),
    ("amino acid metabolic process", [
        "catabolic process", "biosynthetic process",
        "protein modification", "glycoprotein",
        "amyloid", "peptidase", "endopeptidase",
        "protein maturation", "protein processing",
    ]),
    ("signaling", [
        "regulation of", "positive regulation", "negative regulation",
    ]),
]


def ensure_obo(fname):
    fpath = OBO_DIR / fname
    if fpath.exists():
        return fpath
    OBO_DIR.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    for url in OBO_URLS[fname]:
        try:
            print(f"Downloading {fname} from {url}...")
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, context=ctx) as resp:
                data = resp.read()
            with open(fpath, "wb") as f:
                f.write(data)
            print(f"  Saved: {len(data)} bytes")
            return fpath
        except Exception as e:
            print(f"  Failed: {e}")
    print(f"ERROR: Could not download {fname}")
    sys.exit(1)


def load_filtered_gobp_terms():
    """Load GOBP CSVs, filter, return list of (term_name, go_id) tuples."""
    records = []
    prefix = LIBRARY_NAME + "__"
    for lineage in LINEAGES:
        for t1, t2 in DIRECTED_TISSUE_PAIRS:
            label = f"{lineage}_{t1}_vs_{t2}"
            path = INPUT_DIR / f"gsea_{LIBRARY_TAG}_{label}.csv"
            if not path.exists():
                continue
            gsea = pd.read_csv(path)
            gsea["lineage"] = lineage
            gsea["pathway"] = gsea["Term"].str.replace(prefix, "", regex=False)
            records.append(gsea)

    if not records:
        print("ERROR: No GOBP GSEA CSVs found!")
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

    unique_pathways = sorted(passing["pathway"].unique())
    print(f"Total GOBP terms surviving NES/FDR filters: {len(unique_pathways)}")

    results = []
    n_no_id = 0
    for pw in unique_pathways:
        match = GO_ID_RE.search(pw)
        if match:
            go_id = match.group(0)[1:-1]
            results.append((pw, go_id))
        else:
            n_no_id += 1

    if n_no_id > 0:
        print(f"  WARNING: {n_no_id} terms had no GO ID (skipped)")
    print(f"  Terms with GO IDs: {len(results)}")
    return results


def keyword_classify(term_name):
    """Classify a GO term by keyword matching on its name. Returns family or None."""
    t_lower = term_name.lower()
    for family, keywords in KEYWORD_RULES:
        for kw in keywords:
            if kw in t_lower:
                return family
    return None


# ── Signaling sub-family keyword rules (first match wins) ───────────
SIGNALING_SUB_RULES = [
    ("signaling — immune", [
        "immune", "cytokine", "chemokine", "interferon", "interleukin",
        "tnf", "tlr", "tcr", "bcr", "nf-kappa", "nf-kb", "inflammatory",
        "t cell", "b cell", "lymphocyte", "leukocyte", "macrophage",
        "complement", "toll-like", "antigen",
    ]),
    ("signaling — growth factor", [
        "mapk", "pi3k", "egfr", "ras ", "rho ", "rac ", "receptor tyrosine",
        "growth factor", "mtor", "akt", "erbb", "vegf", "pdgf", "fgf",
        "igf", "egf", "kinase cascade", "proliferat",
    ]),
    ("signaling — developmental", [
        "wnt", "notch", "hedgehog", "bmp", "tgf", "hippo", "morphogen",
        "differentiation", "embryo", "pattern", "developmental",
        "stem cell", "fate", "specification",
    ]),
    ("signaling — metabolic", [
        "insulin", "glucose", "lipid", "steroid", "hormone",
        "glucocorticoid", "corticosteroid", "catecholamine",
        "adipocyte", "metabolic",
    ]),
]


def decompose_signaling(term_name):
    """Sub-classify a 'signaling' term into a more specific sub-family."""
    t_lower = term_name.lower()
    for sub_family, keywords in SIGNALING_SUB_RULES:
        for kw in keywords:
            if kw in t_lower:
                return sub_family
    return "signaling — other"


def decompose_signaling_with_ontology(go_id, term_name, dag):
    """Try ontology-based decomposition first, then fall back to keywords.

    Walks ancestors of the term looking for a named ancestor at depth >= 4
    whose name matches a signaling sub-rule keyword. If found, use that
    sub-family. Otherwise fall back to keyword matching on the term itself.
    """
    if go_id in dag:
        # Collect ancestors at depth >= 4, sorted deepest first
        try:
            ancestors = []
            for anc_id in dag[go_id].get_all_parents():
                if anc_id in dag and dag[anc_id].depth >= 4:
                    ancestors.append((dag[anc_id].depth, dag[anc_id].name))
            ancestors.sort(key=lambda x: -x[0])  # deepest first
            for _, anc_name in ancestors:
                sub = decompose_signaling(anc_name)
                if sub != "signaling — other":
                    return sub
        except Exception:
            pass

    return decompose_signaling(term_name)


def main():
    go_obo_path = ensure_obo("go-basic.obo")
    slim_obo_path = ensure_obo("goslim_generic.obo")

    print("Loading GO DAGs...")
    from goatools.obo_parser import GODag
    from goatools.mapslim import mapslim

    dag = GODag(str(go_obo_path))
    slim_dag = GODag(str(slim_obo_path))

    # Build slim name lookup
    slim_names = {}
    for goid, t in slim_dag.items():
        if t.namespace == "biological_process":
            slim_names[goid] = t.name

    terms = load_filtered_gobp_terms()

    # Filter out over-broad terms; terms not in DAG get keyword-classified
    filtered = []
    keyword_only = []
    n_not_in_dag = 0
    n_too_broad = 0
    for pw, go_id in terms:
        if go_id not in dag:
            n_not_in_dag += 1
            keyword_only.append((pw, go_id))  # classify by keyword instead of skipping
            continue
        if dag[go_id].depth <= 2:
            n_too_broad += 1
            continue
        filtered.append((pw, go_id))

    print(f"\nDepth filtering:")
    print(f"  Not in DAG (will keyword-classify): {n_not_in_dag}")
    print(f"  Dropped (depth <= 2): {n_too_broad}")
    print(f"  Remaining for slim-mapping: {len(filtered)}")

    # Phase 1: mapslim (ancestor-based)
    mapped = []
    unmapped_for_keyword = []
    n_slim_mapped = 0

    for pw, go_id in filtered:
        try:
            direct_slims, _ = mapslim(go_id, dag, slim_dag)
        except Exception:
            direct_slims = set()

        # Filter to BP slim terms only
        bp_slims = {s for s in direct_slims if s in slim_names}

        if bp_slims:
            # Pick deepest slim ancestor
            best = max(bp_slims, key=lambda s: dag[s].depth if s in dag else 0)
            family = slim_names[best]
            mapped.append((pw, go_id, family))
            n_slim_mapped += 1
        else:
            unmapped_for_keyword.append((pw, go_id))

    print(f"\nPhase 1 (mapslim): {n_slim_mapped} mapped, "
          f"{len(unmapped_for_keyword)} need keyword fallback")

    # Phase 2: keyword fallback (includes terms not in DAG)
    all_for_keyword = unmapped_for_keyword + keyword_only
    n_keyword_mapped = 0
    still_unmapped = []

    for pw, go_id in all_for_keyword:
        # Strip GO ID from name for keyword matching
        name_only = GO_ID_RE.sub("", pw).strip()
        family = keyword_classify(name_only)
        if family is not None:
            mapped.append((pw, go_id, family))
            n_keyword_mapped += 1
        else:
            still_unmapped.append((pw, go_id))

    print(f"Phase 2 (keyword): {n_keyword_mapped} mapped, "
          f"{len(still_unmapped)} still unmapped")

    if still_unmapped:
        print(f"\nUnmapped terms ({len(still_unmapped)}):")
        for pw, go_id in still_unmapped[:30]:
            print(f"  {pw}")
        if len(still_unmapped) > 30:
            print(f"  ... and {len(still_unmapped) - 30} more")
        # Assign remaining to "other biological process"
        for pw, go_id in still_unmapped:
            mapped.append((pw, go_id, "other biological process"))

    # ── Signaling decomposition ──────────────────────────────────────
    n_signaling_before = sum(1 for _, _, f in mapped if f == "signaling")
    print(f"\nSignaling decomposition: {n_signaling_before} terms in 'signaling'")

    decomposed = []
    sub_counts = Counter()
    for pw, go_id, fam in mapped:
        if fam in ("signaling", "signal transduction"):
            name_only = GO_ID_RE.sub("", pw).strip()
            sub = decompose_signaling_with_ontology(go_id, name_only, dag)
            decomposed.append((pw, go_id, sub))
            sub_counts[sub] += 1
        else:
            decomposed.append((pw, go_id, fam))
    mapped = decomposed

    print(f"  Split into:")
    for sub, cnt in sub_counts.most_common():
        print(f"    {sub}: {cnt}")

    # Consolidate families
    family_counts = Counter(fam for _, _, fam in mapped)
    print(f"\nRaw families: {len(family_counts)}")

    # Merge small families
    large = {name for name, c in family_counts.items() if c >= MIN_FAMILY_SIZE}
    small = {name for name, c in family_counts.items() if c < MIN_FAMILY_SIZE}

    # Keyword-based merge targets
    merge_targets = {
        "carbohydrate metabolic process": ["carbohydrate", "sugar", "glyco"],
        "lipid metabolic process": ["lipid", "fatty"],
        "amino acid metabolic process": ["amino acid", "protein metabol", "proteoly"],
        "generation of precursor metabolites and energy": [
            "metabol", "energy", "oxidat", "redox", "mitochond",
        ],
        "immune system process": ["immune", "defense", "inflam"],
        "cell cycle": ["cell cycle", "mitotic", "division", "chromosome"],
        "signaling — immune": ["immune", "cytokine", "interferon"],
        "signaling — growth factor": ["growth factor", "kinase", "mapk", "ras"],
        "signaling — developmental": ["wnt", "notch", "hedgehog", "morphogen"],
        "signaling — metabolic": ["insulin", "hormone", "steroid"],
        "signaling — other": ["signal", "receptor"],
        "anatomical structure development": ["develop", "morpho", "differenti"],
        "transmembrane transport": ["transport", "channel", "pump"],
        "regulation of DNA-templated transcription": ["transcript", "gene express"],
        "cytoskeleton organization": ["cytoskelet", "actin", "microtubul"],
        "cell motility": ["motil", "migrat", "chemotax"],
    }

    merge_map = {}
    for small_fam in small:
        s_lower = small_fam.lower()
        best = None
        for large_fam in large:
            l_lower = large_fam.lower()
            if s_lower in l_lower or l_lower in s_lower:
                best = large_fam
                break
            for kw in merge_targets.get(large_fam, [l_lower.split()[0]]):
                if kw in s_lower:
                    best = large_fam
                    break
            if best:
                break
        if best:
            merge_map[small_fam] = best

    if merge_map:
        print(f"\nMerges ({len(merge_map)}):")
        for old, new in sorted(merge_map.items()):
            print(f"  '{old}' ({family_counts[old]}) -> '{new}'")

    final_mapped = [
        (pw, go_id, merge_map.get(fam, fam))
        for pw, go_id, fam in mapped
    ]

    # Remaining small -> "other biological process"
    final_counts = Counter(fam for _, _, fam in final_mapped)
    still_small = {n for n, c in final_counts.items() if c < MIN_FAMILY_SIZE}
    if still_small:
        print(f"\nRemaining small families -> 'other biological process' ({len(still_small)}):")
        for s in sorted(still_small):
            print(f"  '{s}' ({final_counts[s]})")
        final_mapped = [
            (pw, go_id, "other biological process" if fam in still_small else fam)
            for pw, go_id, fam in final_mapped
        ]

    def print_distribution(mapped_list, label):
        counts = Counter(fam for _, _, fam in mapped_list)
        print(f"\n{label}: {len(counts)} families, {len(mapped_list)} terms")
        for fam, cnt in counts.most_common():
            print(f"  {fam}: {cnt}")

    # ── Step 1: Collapse metabolism ───────────────────────────────────
    METABOLISM_FAMILIES = {
        "amino acid metabolic process",
        "lipid metabolic process",
        "carbohydrate metabolic process",
        "nucleobase-containing small molecule metabolic process",
        "generation of precursor metabolites and energy",
        "signaling — metabolic",
        "mRNA metabolic process",
    }
    final_mapped = [
        (pw, go_id, "metabolic process" if fam in METABOLISM_FAMILIES else fam)
        for pw, go_id, fam in final_mapped
    ]
    print_distribution(final_mapped, "After step 1 (metabolism merged)")

    # ── Step 2: Decompose immune system process ───────────────────────
    IMMUNE_SUB_RULES = [
        ("immune — antigen presentation", [
            "antigen presentation", "antigen processing", "mhc",
            "peptide loading",
        ]),
        ("immune — effector", [
            "cytotoxic", "killing", "degranulation", "perforin",
            "granzyme", "phagocytosis", "complement activation",
            "respiratory burst",
        ]),
        ("immune — activation", [
            "t cell activation", "b cell activation",
            "lymphocyte activation", "leukocyte activation",
            "t cell differentiation", "b cell differentiation",
            "lymphocyte differentiation",
            "cell activation involved in immune",
        ]),
        ("immune — cytokine", [
            "cytokine production", "interferon production",
            "interleukin", "chemokine production", "tnf production",
        ]),
    ]

    def decompose_immune(term_name):
        t = term_name.lower()
        for sub, kws in IMMUNE_SUB_RULES:
            for kw in kws:
                if kw in t:
                    return sub
        return "immune — other"

    decomposed = []
    for pw, go_id, fam in final_mapped:
        if fam == "immune system process":
            name_only = GO_ID_RE.sub("", pw).strip()
            decomposed.append((pw, go_id, decompose_immune(name_only)))
        else:
            decomposed.append((pw, go_id, fam))
    final_mapped = decomposed
    print_distribution(final_mapped, "After step 2 (immune decomposed)")

    # ── Step 3: Collapse cellular organization ────────────────────────
    CELL_ORG_FAMILIES = {
        "cytoskeleton organization",
        "vesicle-mediated transport",
        "membrane organization",
        "chromatin organization",
        "protein folding",
        "protein-containing complex assembly",
        "mitochondrion organization",
        "microtubule-based movement",
    }
    final_mapped = [
        (pw, go_id, "cellular organization" if fam in CELL_ORG_FAMILIES else fam)
        for pw, go_id, fam in final_mapped
    ]
    print_distribution(final_mapped, "After step 3 (cellular org merged)")

    # ── Step 4: signaling — other → other biological process ─────────
    final_mapped = [
        (pw, go_id,
         "other biological process" if fam == "signaling — other" else fam)
        for pw, go_id, fam in final_mapped
    ]
    print_distribution(final_mapped, "After step 4 (signaling-other collapsed)")

    # ── Step 5: Consolidate small remaining families (threshold 8) ────
    FINAL_MIN_FAMILY_SIZE = 8
    step5_counts = Counter(fam for _, _, fam in final_mapped)
    small_final = {
        n for n, c in step5_counts.items()
        if c < FINAL_MIN_FAMILY_SIZE and n != "other biological process"
    }
    if small_final:
        print(f"\nStep 5: merging {len(small_final)} families "
              f"(<{FINAL_MIN_FAMILY_SIZE} terms) -> 'other biological process':")
        for s in sorted(small_final):
            print(f"  '{s}' ({step5_counts[s]})")
        final_mapped = [
            (pw, go_id,
             "other biological process" if fam in small_final else fam)
            for pw, go_id, fam in final_mapped
        ]
    print_distribution(final_mapped, "After step 5 (small merged)")

    # Write TSV
    tsv_path = MODULES_DIR / "pathway_families_gobp.tsv"
    with open(tsv_path, "w") as f:
        f.write("term\tgo_id\tfamily\n")
        for pw, go_id, fam in sorted(final_mapped):
            f.write(f"{pw}\t{go_id}\t{fam}\n")

    print(f"\nWrote {len(final_mapped)} terms to {tsv_path}")

    final_counts = Counter(fam for _, _, fam in final_mapped)
    print(f"\nFamily distribution ({len(final_counts)} families, "
          f"{len(final_mapped)} terms):")
    for fam, count in final_counts.most_common():
        print(f"  {fam}: {count}")


if __name__ == "__main__":
    main()
