import io
import re
import string
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, process

st.set_page_config(page_title="Consumables Master Data Harmonization", layout="wide")

# ═══════════════════════════════════════════════════════════════════
#  DICTIONARIES & PATTERNS
# ═══════════════════════════════════════════════════════════════════

# Medical suture brand names → generic chemical names used in NPC
SUTURE_BRANDS: Dict[str, str] = {
    # Covidien / Medtronic
    "polysorb":      "coated polyglactin 910",
    "velosorb":      "polyglactin 910",
    "polycryl":      "polyglactin 910",
    "biosyn":        "glycomer 631 absorbable suture",
    "caprosyn":      "polyglytone 6211 absorbable suture",
    "surgipro":      "polypropylene suture",
    "surgiproii":    "polypropylene suture",
    "ticron":        "braided polyester suture",
    "vloc":          "barbed absorbable suture",
    "monosof":       "nylon monofilament suture",
    "novafil":       "polybutester suture",
    # Ethicon / J&J
    "vicryl":        "coated polyglactin 910",
    "vicrylrapide":  "coated polyglactin 910 rapid",
    "pds":           "polydioxanone suture",
    "prolene":       "polypropylene suture",
    "ethilon":       "nylon suture",
    "maxon":         "polyglyconate suture",
    "mersilene":     "polyester suture",
    "ethibond":      "polyester suture",
    "surgilon":      "braided nylon suture",
    "nurolon":       "braided nylon suture",
    # Generic synonyms
    "injector":      "syringe",
    "hand gloves":   "gloves",
    "iv line":       "cannula",
}

# Word-level synonym replacements (applied token by token)
WORD_SYNONYMS: Dict[str, str] = {
    "cortex":      "cortical",
    "concelous":   "cancellous",
    "cancellus":   "cancellous",
    "concelouos":  "cancellous",
    "injector":    "syringe",
    "milliliter":  "ml",
    "millilitre":  "ml",
    "millimeter":  "mm",
    "millimetre":  "mm",
}

CORE_KEYWORDS = {
    "syringe", "catheter", "screw", "gloves", "cannula",
    "needle", "tube", "mask", "bandage", "gauze", "suture",
    "plate", "nail", "drain", "trocar", "stapler", "clip",
    "electrode", "probe", "forceps", "scissors", "retractor",
    "cannula", "introducer", "sheath",
}

# Patterns to strip before matching (noise / packaging / brand qualifiers)
NOISE_PATTERNS: List[re.Pattern] = [
    re.compile(r'\(pitch[^)]*\)',          re.IGNORECASE),  # (PITCH 1.25)
    re.compile(r'\bpitch\s*[\d.,]*',       re.IGNORECASE),  # PITCH 1.25 standalone
    re.compile(r'\bself[-\s]*tapp?ing\b',  re.IGNORECASE),  # SELF TAPPING
    re.compile(r'\b[sS]\.[tT]\.?\b'),                       # S.T / S.T.
    re.compile(r'\bvio\b',                 re.IGNORECASE),  # VIO (violet colour)
    re.compile(r'\bdlu\b',                 re.IGNORECASE),  # DLU (packaging code)
    re.compile(r'\bx\s*\d+\b(?!\s*mm)',   re.IGNORECASE),  # X6/X12 pack counts
    re.compile(r'\b\d+\s*x\s*\d+[a-z]*\b',re.IGNORECASE), # 5x45cm (dimensions-as-pack)
    re.compile(r'\bdt\b',                  re.IGNORECASE),  # DT
    re.compile(r'\bgs\d+\b',              re.IGNORECASE),  # GS21, GS25
    re.compile(r'\bpremium\b',            re.IGNORECASE),  # PREMIUM qualifier
    re.compile(r'\bbb\d+\b',              re.IGNORECASE),  # BB26 needle codes
    re.compile(r'\b[a-z]{1,2}\d{2}\b',   re.IGNORECASE),  # ST4, V20, etc.
    re.compile(r'\bdisp[oo]?\b',          re.IGNORECASE),  # DISP / DISPO
    re.compile(r'\bassy\b',               re.IGNORECASE),  # ASSY
    re.compile(r'\bsterile\b',            re.IGNORECASE),  # STERILE
    re.compile(r'\buu\b',                 re.IGNORECASE),  # UU (single use)
    re.compile(r'\bsingle\s*use\b',       re.IGNORECASE),  # SINGLE USE
    re.compile(r'\*'),                                       # Brand asterisk
]

DIM_X_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)(mm)x(\d+(?:\.\d+)?)(mm)', re.IGNORECASE
)
UNIT_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(mm|ml|cc|fr|fg|ch|g|mg|mcg)\b', re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MatchResult:
    npc_code: Optional[str]
    rw_description: Optional[str]
    ihbs_code: Optional[str]
    match_source: str
    match_score: float
    confidence_level: str
    candidate_desc: Optional[str]
    duplicate_flag: bool
    review_notes: str


# ═══════════════════════════════════════════════════════════════════
#  NORMALISATION ENGINE
# ═══════════════════════════════════════════════════════════════════

def normalize_text(text: object, aggressive: bool = True) -> str:
    """
    Full normalisation pipeline:
      1. DxL dimension format expansion (3.5MMx20MM → 3.5 MM 20 MM)
      2. Noise pattern removal (packaging codes, brand qualifiers, pitch info)
      3. Unit spelling normalization
      4. Suture brand name → generic chemical name substitution
      5. Word-level synonym replacement
      6. Punctuation removal (decimal points in numbers preserved)
    """
    if pd.isna(text):
        return ""
    value = str(text).lower().strip()

    # 1. Expand DxL dimension format BEFORE punctuation removal
    value = DIM_X_PATTERN.sub(
        lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)}", value
    )

    # 2. Strip noise patterns (while punctuation still present for accurate matching)
    if aggressive:
        for pattern in NOISE_PATTERNS:
            value = pattern.sub(" ", value)

    # 3. Unit spelling
    value = (value
             .replace("milliliter", "ml").replace("millilitre", "ml")
             .replace("millimeter", "mm").replace("millimetre", "mm"))

    # 4. Brand → generic (multi-word phrases first)
    for brand, generic in SUTURE_BRANDS.items():
        value = re.sub(r"\b" + re.escape(brand) + r"\b", generic, value, flags=re.IGNORECASE)

    # 5. Word-level synonyms
    tokens = value.split()
    tokens = [WORD_SYNONYMS.get(t, t) for t in tokens]
    value = " ".join(tokens)

    # 6. Remove punctuation but PRESERVE decimal points between digits
    value = re.sub(r"(?<=\d)\.(?=\d)", "DEC", value)  # protect 3.5 → 3DEC5
    value = re.sub(r"[^\w\s]", " ", value)
    value = value.replace("DEC", ".")                   # restore 3DEC5 → 3.5

    return re.sub(r"\s+", " ", value).strip()


def extract_size_tokens(text: str) -> List[Tuple[float, str]]:
    return [(float(v), u.lower()) for v, u in UNIT_PATTERN.findall(text)]


def core_keyword(text: str) -> Optional[str]:
    tokens = set(text.split())
    for kw in CORE_KEYWORDS:
        if kw in tokens:
            return kw
    return None


def word_count(text: str) -> int:
    return len([t for t in text.split() if len(t) > 1])


# ═══════════════════════════════════════════════════════════════════
#  SCORING & PENALTY
# ═══════════════════════════════════════════════════════════════════

def multi_score(query: str, candidate: str) -> float:
    """Multi-scorer ensemble — avoids false positives from any single scorer."""
    if not query or not candidate:
        return 0.0
    return max(
        fuzz.token_set_ratio(query, candidate),
        fuzz.token_sort_ratio(query, candidate),
        fuzz.WRatio(query, candidate),
    )


def consistency_penalty(source_text: str, candidate_text: str) -> Tuple[float, List[str]]:
    """
    Penalise mismatches in size (numeric dimensions) and product category.
    Also penalises very short NPC descriptions to prevent false positives
    against generic catch-all entries like 'SUTURE KIT'.
    """
    penalty = 0.0
    notes: List[str] = []

    # Guard against matching very short / generic NPC descriptions
    if word_count(candidate_text) <= 3:
        penalty += 15
        notes.append("Short NPC description (possible generic entry)")

    # Size consistency
    src_sizes = extract_size_tokens(source_text)
    cand_sizes = extract_size_tokens(candidate_text)
    if src_sizes and cand_sizes:
        if not set(src_sizes).intersection(set(cand_sizes)):
            src_units  = {u for _, u in src_sizes}
            cand_units = {u for _, u in cand_sizes}
            penalty += 12 if src_units == cand_units else 15
            notes.append("Size mismatch penalty applied")

    # Core keyword consistency
    src_kw  = core_keyword(source_text)
    cand_kw = core_keyword(candidate_text)
    if src_kw and cand_kw and src_kw != cand_kw:
        penalty += 20
        notes.append(f"Keyword mismatch ({src_kw} vs {cand_kw})")

    return penalty, notes


def confidence_label(score: float) -> str:
    if score >= 80:
        return "HIGH"
    if score >= 65:
        return "MEDIUM"
    return "LOW"


def best_match(
    query: str,
    reference_norm: List[str],
    min_score: float,
    limit: int = 5,
) -> Optional[Tuple[int, float]]:
    """
    Find the best matching candidate:
      - Pre-filter with fast token_set_ratio
      - Rescore top candidates with multi-scorer ensemble
      - Return (index, score) or None
    """
    if not query:
        return None

    result = process.extract(query, reference_norm, scorer=fuzz.token_set_ratio, limit=limit)
    if not result:
        return None

    rescored = [
        (idx, multi_score(query, text))
        for text, base_score, idx in result
        if base_score >= min_score - 15
    ]

    if not rescored:
        top = result[0]
        return (top[2], float(top[1])) if top[1] >= min_score else None

    rescored.sort(key=lambda x: -x[1])
    top_idx, top_score = rescored[0]
    return (top_idx, float(top_score)) if top_score >= min_score else None


def _structural_query(norm_text: str) -> Optional[str]:
    """
    Reduce a description to its core matchable structure:
    <core_keyword> <primary_dimension> [<secondary_dimension>]
    Used as a relaxed fallback when full-text matching fails.
    """
    kw = core_keyword(norm_text)
    sizes = extract_size_tokens(norm_text)
    parts = []
    if kw:
        parts.append(kw)
    for v, u in sizes[:2]:
        parts.append(f"{v:g} {u}")
    return " ".join(parts) if len(parts) >= 2 else None


# ═══════════════════════════════════════════════════════════════════
#  MATCHING ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def match_ihs_row(
    ihs_desc: str,
    npc_df: pd.DataFrame,
    phc_df: pd.DataFrame,
    npc_norm_col: str,
    phc_norm_col: str,
    npc_code_col: str,
    npc_desc_col: str,
    phc_code_col: str,
    phc_desc_col: str,
    phc_ihbs_col: str,
) -> MatchResult:

    npc_norm = npc_df[npc_norm_col].tolist()
    phc_norm = phc_df[phc_norm_col].tolist()
    review_notes: List[str] = []

    def _build_result(df, idx, score, notes, source, ihbs=None) -> MatchResult:
        return MatchResult(
            npc_code=str(df.iloc[idx][npc_code_col if source != "PHC" else phc_code_col]),
            rw_description=str(df.iloc[idx][npc_desc_col if source != "PHC" else phc_desc_col]),
            ihbs_code=ihbs,
            match_source=source,
            match_score=round(score, 2),
            confidence_level=confidence_label(score),
            candidate_desc=str(df.iloc[idx][npc_desc_col if source != "PHC" else phc_desc_col]),
            duplicate_flag=False,
            review_notes="; ".join(notes),
        )

    # ── Step 1: Primary match against NPC (threshold 65) ──
    primary = best_match(ihs_desc, npc_norm, min_score=65)
    if primary:
        idx, score = primary
        penalty, notes = consistency_penalty(ihs_desc, npc_norm[idx])
        final = score - penalty
        review_notes.extend(notes)
        if final >= 65:
            return _build_result(npc_df, idx, final, review_notes, "NPC")

    # ── Step 2: Fallback match against PHC (threshold 60) ──
    secondary = best_match(ihs_desc, phc_norm, min_score=60)
    if secondary:
        idx, score = secondary
        penalty, notes = consistency_penalty(ihs_desc, phc_norm[idx])
        final = score - penalty
        review_notes.extend(notes)
        if final >= 60:
            return _build_result(
                phc_df, idx, final, review_notes, "PHC",
                ihbs=str(phc_df.iloc[idx][phc_ihbs_col]),
            )

    # ── Step 3: Structural fallback against NPC (keyword + dimension only) ──
    structural = _structural_query(ihs_desc)
    if structural and structural != ihs_desc:
        relaxed = best_match(structural, npc_norm, min_score=60)
        if relaxed:
            idx, score = relaxed
            penalty, notes = consistency_penalty(structural, npc_norm[idx])
            final = score - penalty
            review_notes.extend(notes)
            review_notes.append("Structural fallback match")
            if final >= 60:
                return _build_result(npc_df, idx, final, review_notes, "NPC_STRUCTURAL")

    return MatchResult(
        npc_code=None, rw_description=None, ihbs_code=None,
        match_source="UNMATCHED", match_score=0.0,
        confidence_level="LOW", candidate_desc=None,
        duplicate_flag=False, review_notes="No suitable match found",
    )


# ═══════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="IHS_Enriched")
    return output.getvalue()


def mapping_ui(df: pd.DataFrame, label: str, keys: Dict[str, str]) -> Dict[str, str]:
    st.subheader(label)
    mapping: Dict[str, str] = {}
    cols = [""] + df.columns.tolist()
    for human_name, map_key in keys.items():
        mapping[map_key] = st.selectbox(
            f"{label}: {human_name}", cols, key=f"{label}-{map_key}"
        )
    return mapping


# ═══════════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    st.title("Consumables Master Data Harmonization  (IHS ↔ NPC ↔ PHC)")
    st.caption("v2 — Improved matching engine with brand-name mapping, noise removal, and multi-scorer ensemble")

    with st.expander("ℹ️  What changed in v2?", expanded=False):
        st.markdown("""
| Improvement | Details |
|---|---|
| 🏷️ **Suture brand mapping** | 20+ brand names (POLYSORB, BIOSYN, TICRON, VLOC, CAPROSYN, SURGIPRO, VICRYL …) replaced with their generic chemical names before matching |
| 🔤 **Terminology synonyms** | `CORTEX` → `CORTICAL`, spelling variants for cancellous, unit spellings etc. |
| 🔧 **Dimension format** | `3.5MMx20MM` expanded to `3.5 MM 20 MM` so sizes are matched correctly |
| 🗑️ **Noise removal** | Strips packaging codes (DLU, X6, VIO, GS21, ST4), pitch info `(PITCH 1.25)`, self-tapping abbreviations before matching |
| 📊 **Multi-scorer ensemble** | `token_set_ratio` + `token_sort_ratio` + `WRatio` — best score wins |
| 🔄 **Structural fallback** | If full-text match fails, tries matching on core keyword + primary dimension only |
| 🛡️ **False-positive guards** | Short NPC descriptions (e.g. "SUTURE KIT") penalised; size-unit mismatches penalised more aggressively |
        """)

    # ── File upload ──
    c1, c2, c3 = st.columns(3)
    with c1:
        ihs_file = st.file_uploader("IHS File", type=["xlsx", "xls"], key="ihs")
    with c2:
        npc_file = st.file_uploader("NPC File", type=["xlsx", "xls"], key="npc")
    with c3:
        phc_file = st.file_uploader("PHC File", type=["xlsx", "xls"], key="phc")

    if not (ihs_file and npc_file and phc_file):
        st.info("Upload all three Excel files to continue.")
        return

    ihs_df = pd.read_excel(ihs_file)
    npc_df = pd.read_excel(npc_file)
    phc_df = pd.read_excel(phc_file)

    # ── Preview ──
    st.markdown("### Dataset Preview")
    tab1, tab2, tab3 = st.tabs(["IHS", "NPC", "PHC"])
    with tab1:
        st.dataframe(ihs_df.head(10), use_container_width=True)
    with tab2:
        st.dataframe(npc_df.head(10), use_container_width=True)
    with tab3:
        st.dataframe(phc_df.head(10), use_container_width=True)

    # ── Column mapping ──
    st.markdown("### Column Mapping")
    ihs_map = mapping_ui(ihs_df, "IHS", {"Product Name / Description": "desc"})
    npc_map = mapping_ui(npc_df, "NPC", {
        "NPC Code": "npc_code",
        "RW Product Description": "rw_desc",
    })
    phc_map = mapping_ui(phc_df, "PHC", {
        "Product Description": "desc",
        "NPC Code": "npc_code",
        "IHBS Code": "ihbs_code",
    })

    # ── Options ──
    col_a, col_b = st.columns(2)
    with col_a:
        aggressive = st.checkbox(
            "Aggressive noise removal",
            value=True,
            help="Strips packaging codes, colour indicators, and abbreviated qualifiers. Recommended ON.",
        )
    with col_b:
        show_unmatched = st.checkbox(
            "Show unmatched items after run",
            value=False,
            help="Display the first 50 unmatched IHS descriptions after matching.",
        )

    # ── Run ──
    if st.button("▶  Run Matching", type="primary"):
        all_maps = [
            ihs_map["desc"], npc_map["npc_code"], npc_map["rw_desc"],
            phc_map["desc"], phc_map["npc_code"], phc_map["ihbs_code"],
        ]
        if any(not m for m in all_maps):
            st.error("Please map all required columns before running matching.")
            return

        progress    = st.progress(0)
        status_text = st.empty()

        # Normalise
        status_text.text("Normalising descriptions…")
        ihs_work = ihs_df.copy()
        npc_work = npc_df.copy()
        phc_work = phc_df.copy()
        ihs_work["_norm"] = ihs_work[ihs_map["desc"]].apply(lambda x: normalize_text(x, aggressive))
        npc_work["_norm"] = npc_work[npc_map["rw_desc"]].apply(lambda x: normalize_text(x, aggressive))
        phc_work["_norm"] = phc_work[phc_map["desc"]].apply(lambda x: normalize_text(x, aggressive))

        # Match
        results = []
        total = len(ihs_work)
        for i, (_, row) in enumerate(ihs_work.iterrows()):
            match = match_ihs_row(
                ihs_desc      = row["_norm"],
                npc_df        = npc_work,
                phc_df        = phc_work,
                npc_norm_col  = "_norm",
                phc_norm_col  = "_norm",
                npc_code_col  = npc_map["npc_code"],
                npc_desc_col  = npc_map["rw_desc"],
                phc_code_col  = phc_map["npc_code"],
                phc_desc_col  = phc_map["desc"],
                phc_ihbs_col  = phc_map["ihbs_code"],
            )
            results.append(match)
            if i % 50 == 0:
                progress.progress(min((i + 1) / max(total, 1), 1.0))
                status_text.text(f"Matching {i + 1} / {total}…")

        progress.progress(1.0)
        status_text.text("Complete!")

        # Build output dataframe
        result_df = ihs_work.drop(columns=["_norm"]).copy()
        result_df["NPC_Code"]                     = [r.npc_code        for r in results]
        result_df["RW_Product_Description"]       = [r.rw_description  for r in results]
        result_df["IHBS_Code"]                    = [r.ihbs_code       for r in results]
        result_df["Match_Source"]                 = [r.match_source    for r in results]
        result_df["Match_Score"]                  = [r.match_score     for r in results]
        result_df["Confidence_Level"]             = [r.confidence_level for r in results]
        result_df["Matched_Candidate_Description"]= [r.candidate_desc  for r in results]
        result_df["Duplicate_Match_Flag"]         = [r.duplicate_flag  for r in results]
        result_df["Review_Notes"]                 = [r.review_notes    for r in results]

        # ── Summary dashboard ──
        src_counts = result_df["Match_Source"].value_counts()
        matched    = (result_df["Match_Source"] != "UNMATCHED").sum()
        rate       = matched / len(result_df) * 100 if len(result_df) else 0

        st.success(f"✅ **{rate:.1f}%** of IHS items matched  ({matched} / {len(result_df)})")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("NPC (direct)",       src_counts.get("NPC", 0))
        m2.metric("NPC (structural)",   src_counts.get("NPC_STRUCTURAL", 0))
        m3.metric("PHC fallback",       src_counts.get("PHC", 0))
        m4.metric("Unmatched",          src_counts.get("UNMATCHED", 0))
        conf = result_df[result_df["Match_Source"] != "UNMATCHED"]["Confidence_Level"].value_counts()
        m5.metric("HIGH confidence",    conf.get("HIGH", 0))

        c1, c2, c3 = st.columns(3)
        c1.metric("HIGH",   conf.get("HIGH", 0))
        c2.metric("MEDIUM", conf.get("MEDIUM", 0))
        c3.metric("LOW",    conf.get("LOW", 0))

        if show_unmatched:
            st.markdown("#### Unmatched items (review for further synonym additions)")
            unmatched_df = result_df[result_df["Match_Source"] == "UNMATCHED"][[ihs_map["desc"]]].reset_index(drop=True)
            st.dataframe(unmatched_df.head(50), use_container_width=True)

        st.markdown("### Enriched Output Preview")
        st.dataframe(result_df.head(100), use_container_width=True)

        st.download_button(
            "⬇️  Download IHS_Enriched_With_NPC.xlsx",
            data=to_excel_bytes(result_df),
            file_name="IHS_Enriched_With_NPC.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
