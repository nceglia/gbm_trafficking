"""Build pathway_families_hallmark.tsv from the gseapy Hallmark library.

Maps every MSigDB Hallmark 2020 term to one of 12 pathway families
defined in modules.style.PATHWAY_FAMILY_ORDER.  Fails loudly if any
term is unmapped or if the library gains/loses terms.

Usage:
    conda run -n scrna python pipeline/scripts/build_hallmark_family_map.py
"""
import sys
from pathlib import Path

import gseapy as gp

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))
from modules.style import PATHWAY_FAMILY_ORDER

# ── Hardcoded term → family mapping (exact gseapy names) ─────────────
TERM_TO_FAMILY = {
    # NF-kB / Inflammation
    "TNF-alpha Signaling via NF-kB": "NF-kB / Inflammation",
    "Inflammatory Response":         "NF-kB / Inflammation",
    "Complement":                    "NF-kB / Inflammation",
    "Allograft Rejection":           "NF-kB / Inflammation",
    "Coagulation":                   "NF-kB / Inflammation",
    # Stress response
    "Hypoxia":                          "Stress response",
    "UV Response Up":                   "Stress response",
    "UV Response Dn":                   "Stress response",
    "Unfolded Protein Response":        "Stress response",
    "p53 Pathway":                      "Stress response",
    "Reactive Oxygen Species Pathway":  "Stress response",
    # Apoptosis / cell death
    "Apoptosis": "Apoptosis / cell death",
    # Proliferation / cell cycle
    "E2F Targets":    "Proliferation / cell cycle",
    "G2-M Checkpoint": "Proliferation / cell cycle",
    "Mitotic Spindle": "Proliferation / cell cycle",
    "Myc Targets V1":  "Proliferation / cell cycle",
    "Myc Targets V2":  "Proliferation / cell cycle",
    # DNA damage / repair
    "DNA Repair": "DNA damage / repair",
    # Type I/II interferon
    "Interferon Alpha Response": "Type I/II interferon",
    "Interferon Gamma Response": "Type I/II interferon",
    # Cytokine signaling
    "IL-2/STAT5 Signaling":     "Cytokine signaling",
    "IL-6/JAK/STAT3 Signaling": "Cytokine signaling",
    # TCR / antigen receptor — none in Hallmark
    # Metabolism — oxidative
    "Oxidative Phosphorylation": "Metabolism — oxidative",
    "Fatty Acid Metabolism":     "Metabolism — oxidative",
    "Adipogenesis":              "Metabolism — oxidative",
    "Bile Acid Metabolism":      "Metabolism — oxidative",
    "Xenobiotic Metabolism":     "Metabolism — oxidative",
    "Pperoxisome":               "Metabolism — oxidative",
    # Metabolism — biosynthetic
    "Glycolysis":              "Metabolism — biosynthetic",
    "mTORC1 Signaling":        "Metabolism — biosynthetic",
    "Cholesterol Homeostasis": "Metabolism — biosynthetic",
    "heme Metabolism":         "Metabolism — biosynthetic",
    "Protein Secretion":       "Metabolism — biosynthetic",
    # Migration / adhesion
    "Apical Junction": "Migration / adhesion",
    "Apical Surface":  "Migration / adhesion",
    # Growth factor signaling
    "KRAS Signaling Up":          "Growth factor signaling",
    "KRAS Signaling Dn":          "Growth factor signaling",
    "PI3K/AKT/mTOR  Signaling":   "Growth factor signaling",
    "TGF-beta Signaling":         "Growth factor signaling",
    "Wnt-beta Catenin Signaling": "Growth factor signaling",
    "Notch Signaling":            "Growth factor signaling",
    "Hedgehog Signaling":         "Growth factor signaling",
    # Hormone response
    "Estrogen Response Early": "Hormone response",
    "Estrogen Response Late":  "Hormone response",
    "Androgen Response":       "Hormone response",
    # Tissue morphogenesis
    "Epithelial Mesenchymal Transition": "Tissue morphogenesis",
    "Angiogenesis":                      "Tissue morphogenesis",
    "Myogenesis":                        "Tissue morphogenesis",
    "Spermatogenesis":                   "Tissue morphogenesis",
    "Pancreas Beta Cells":               "Tissue morphogenesis",
}


def main():
    # 1. Get library terms from gseapy
    lib = gp.get_library("MSigDB_Hallmark_2020")
    gseapy_terms = set(lib.keys())
    mapped_terms = set(TERM_TO_FAMILY.keys())

    # 2. Validate: no missing, no extra
    in_lib_not_mapped = gseapy_terms - mapped_terms
    in_map_not_lib = mapped_terms - gseapy_terms

    errors = []
    if in_lib_not_mapped:
        errors.append(
            f"Terms in gseapy but NOT in mapping dict ({len(in_lib_not_mapped)}):\n"
            + "\n".join(f"  {t!r}" for t in sorted(in_lib_not_mapped))
        )
    if in_map_not_lib:
        errors.append(
            f"Terms in mapping dict but NOT in gseapy ({len(in_map_not_lib)}):\n"
            + "\n".join(f"  {t!r}" for t in sorted(in_map_not_lib))
        )

    # Validate all families are legal
    bad_families = {f for f in TERM_TO_FAMILY.values() if f not in PATHWAY_FAMILY_ORDER}
    if bad_families:
        errors.append(f"Unknown families: {bad_families}")

    if errors:
        print("FATAL: Hallmark family mapping validation failed!\n")
        print("\n".join(errors))
        sys.exit(1)

    # 3. Write TSV
    out_path = REPO_ROOT / "pipeline" / "modules" / "pathway_families_hallmark.tsv"
    with open(out_path, "w") as f:
        f.write("term\tfamily\n")
        for term in sorted(TERM_TO_FAMILY.keys()):
            f.write(f"{term}\t{TERM_TO_FAMILY[term]}\n")

    print(f"Wrote {len(TERM_TO_FAMILY)} terms to {out_path}")
    print(f"  Families represented: {len(set(TERM_TO_FAMILY.values()))}/12")

    # Summary
    from collections import Counter
    counts = Counter(TERM_TO_FAMILY.values())
    for fam in PATHWAY_FAMILY_ORDER:
        n = counts.get(fam, 0)
        print(f"  {fam}: {n}")


if __name__ == "__main__":
    main()
