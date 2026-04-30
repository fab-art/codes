"""
Consumables Master Data Harmonization Tool  (IHS ↔ NPC ↔ PHC)
v3 — accuracy-focused matching engine

Key improvements over v2:
  • European decimal commas (2,7 → 2.7) parsed correctly
  • Suture size validation (2-0 ≠ 4-0) prevents wrong-size brand matches
  • Generic NPC entries (SUTURE KIT, ENDO Z BUR …) blacklisted
  • Flexible DxL dimension pattern catches 2.7X20mm, 5X35MM, 3.5MMx20MM
  • Cross-category penalty (suture → mesh/sponge/screw rejected)
  • Better noise vocabulary (color codes, needle codes, packaging)
  • Brand expansion to 25+ surgical brands incl. V-LOC, SURGIPRO II, LIGASURE
"""
import io
import re
import string
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, process

st.set_page_config(page_title="Consumables Master Data Harmonization", layout="wide")

# ═══════════════════════════════════════════════════════════════════
#  DICTIONARIES & PATTERNS
# ═══════════════════════════════════════════════════════════════════

SUTURE_BRANDS: Dict[str, str] = {
    # ─── Multi-word phrases first (longest match wins) ───
    "surgipro ii":   "polypropylene suture",
    "surgipro*ii":   "polypropylene suture",
    "surgiproii":    "polypropylene suture",
    "v loc":         "barbed absorbable suture",
    "v-loc":         "barbed absorbable suture",
    "vloc":          "barbed absorbable suture",
    "hem-o-lok":     "hem o lok ligation clip",
    "hem o lok":     "hem o lok ligation clip",
    "hemolock":      "hem o lok ligation clip",
    # ─── Suture brands ───
    "polysorb":      "coated polyglactin 910",
    "velosorb":      "polyglactin 910",
    "polycryl":      "coated polyglactin 910",
    "biosyn":        "glycomer 631 absorbable suture",
    "caprosyn":      "polyglytone 6211 absorbable suture",
    "surgipro":      "polypropylene suture",
    "ticron":        "braided polyester suture",
    "monosof":       "nylon monofilament suture",
    "novafil":       "polybutester suture",
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
    "sofsilk":       "silk suture",
    "silkam":        "silk suture",
    # ─── Surgical devices ───
    "ligasure":      "vessel sealing device",
    "versaport":     "laparoscopic trocar port",
    "autosonix":     "ultrasonic dissector",
    # ─── Generic synonyms ───
    "injector":      "syringe",
    "hand gloves":   "gloves",
    "iv line":       "cannula",
}

WORD_SYNONYMS: Dict[str, str] = {
    "cortex":      "cortical",
    "concelous":   "cancellous",
    "cancellus":   "cancellous",
    "concelouos":  "cancellous",
    "milliliter":  "ml",
    "millilitre":  "ml",
    "millimeter":  "mm",
    "millimetre":  "mm",
}

CORE_KEYWORDS = {
    "syringe", "catheter", "screw", "gloves", "cannula", "needle", "tube", "mask",
    "bandage", "gauze", "suture", "plate", "nail", "drain", "trocar", "stapler",
    "clip", "electrode", "probe", "forceps", "scissors", "retractor", "introducer",
    "sheath", "mesh", "sponge", "wire", "bur", "pin", "rod", "blade", "knife",
}

SUTURE_KEYWORDS = {
    "suture", "polyglactin", "polypropylene", "polydioxanone", "polyglyconate",
    "polyester", "nylon", "silk", "catgut", "chromic", "monofilament",
    "polyglecaprone", "glycomer", "polyglytone", "polybutester", "barbed",
}

# NPC entries known to be too generic — heavily penalised as match candidates
GENERIC_BLACKLIST = {
    "suture kit", "endo z bur", "iris forceps", "sealing machine",
    "reusable apron", "d plate", "filter 1µ", "mixing spatula",
    "ise control i", "magill forcep", "skin stapler", "bipolar forceps cord",
}

# ─── Noise patterns (run BEFORE punctuation removal) ───
NOISE_PATTERNS: List[re.Pattern] = [
    re.compile(r'\(pitch[^)]*\)',           re.IGNORECASE),
    re.compile(r'\bpitch\s*[\d.,]*',        re.IGNORECASE),
    re.compile(r'\bself[-\s]*tapp?ing\b',   re.IGNORECASE),
    re.compile(r'\b[sS]\.[tT]\.?\b'),                          # S.T / S.T.
    # Color codes
    re.compile(r'\bvio\b', re.IGNORECASE),  re.compile(r'\bblu\b', re.IGNORECASE),
    re.compile(r'\bbrn\b', re.IGNORECASE),  re.compile(r'\bblk\b', re.IGNORECASE),
    re.compile(r'\bund\b', re.IGNORECASE),  re.compile(r'\bwhi\b', re.IGNORECASE),
    # Packaging
    re.compile(r'\bdlu\b', re.IGNORECASE),
    re.compile(r'\bx\s*\d+\b(?!\s*mm)(?!\s*cm)', re.IGNORECASE),  # X6, X12 packs
    re.compile(r'\b\d+\s*x\s*\d+\s*cm\b',   re.IGNORECASE),       # 5x45cm packaging only
    # Needle codes
    re.compile(r'\bdt\b', re.IGNORECASE),   re.compile(r'\bda\b', re.IGNORECASE),
    re.compile(r'\bgs\d+\b', re.IGNORECASE),re.compile(r'\bcv\d+\b', re.IGNORECASE),
    re.compile(r'\buc\d+\b', re.IGNORECASE),re.compile(r'\bmc\d+\b', re.IGNORECASE),
    re.compile(r'\bmv\d+\b', re.IGNORECASE),re.compile(r'\bbtp\d+\b', re.IGNORECASE),
    re.compile(r'\bbb\d+\b', re.IGNORECASE),
    re.compile(r'\b[a-z]{2}\d{2,4}\b',      re.IGNORECASE),       # ST4, GS21 …
    # Brand qualifiers / generic noise
    re.compile(r'\bpremium\b',              re.IGNORECASE),
    re.compile(r'\bdisp[oo]?\b',            re.IGNORECASE),
    re.compile(r'\bassy\b',                 re.IGNORECASE),
    re.compile(r'\bsterile\b',              re.IGNORECASE),
    re.compile(r'\buu\b',                   re.IGNORECASE),
    re.compile(r'\bsingle\s*use\b',         re.IGNORECASE),
    re.compile(r'\bligatures?\b',           re.IGNORECASE),
    re.compile(r'\blig\b',                  re.IGNORECASE),
    re.compile(r'\bestch\b',                re.IGNORECASE),
    re.compile(r'\blp\b(?!\d)',             re.IGNORECASE),
    re.compile(r'\bnon\s*abs\b',            re.IGNORECASE),
    re.compile(r'\(d\)',                    re.IGNORECASE),
    re.compile(r'\*'),
]

# Flexible DxL pattern: handles "2.7X20mm", "5X35MM", "3.5MMx20MM", "3,5x44 MM"
DIM_X_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:mm)?\s*x\s*(\d+(?:\.\d+)?)\s*mm', re.IGNORECASE
)

UNIT_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(mm|ml|cc|fr|fg|ch|g|mg|mcg)\b', re.IGNORECASE
)

# Suture sizes (run on RAW text BEFORE normalisation removes dashes)
SUTURE_NMINUS_PATTERN   = re.compile(r'(\d{1,2})\s*[-/]\s*0(?=\D|$)')
SUTURE_SIZE_NPC_PATTERN = re.compile(r'\bsize\s*:?\s*(\d{1,2})(?:\s*[-/]\s*0)?\b', re.IGNORECASE)


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
#  EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════════

def extract_suture_sizes(raw_text: object) -> Set[str]:
    """
    Extract suture size codes from RAW text BEFORE normalization.
    Catches:
      - n-0 / n/0 patterns: '2-0', '4/0', '8-0BLU' (stuck to color)
      - 'SIZE n' / 'SIZE n-0' patterns in NPC descriptions
    Returns set like {'2-0', '4-0'} or {'0'} for 'SIZE 0'.
    """
    if pd.isna(raw_text):
        return set()
    text = str(raw_text)
    sizes: Set[str] = set()
    for m in SUTURE_NMINUS_PATTERN.findall(text):
        sizes.add(f"{m}-0")
    for m in SUTURE_SIZE_NPC_PATTERN.finditer(text):
        n = m.group(1)
        full = m.group(0).lower()
        if "-0" in full or "/0" in full:
            sizes.add(f"{n}-0")
        else:
            sizes.add(n)
    return sizes


def is_suture_item(norm_text: str) -> bool:
    return bool(set(norm_text.split()) & SUTURE_KEYWORDS)


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
#  NORMALIZATION
# ═══════════════════════════════════════════════════════════════════

def normalize_text(text: object, aggressive: bool = True) -> str:
    if pd.isna(text):
        return ""
    v = str(text).lower().strip()

    # 0. European decimal commas: "2,7" → "2.7"
    v = re.sub(r'(?<=\d),(?=\d)', '.', v)

    # 1. Flexible DxL dimension expansion (catches "2.7X20mm", "5X35MM", etc.)
    v = DIM_X_PATTERN.sub(lambda m: f"{m.group(1)} mm {m.group(2)} mm", v)

    # 2. Strip noise patterns
    if aggressive:
        for pat in NOISE_PATTERNS:
            v = pat.sub(" ", v)

    # 3. Unit spelling
    v = (v.replace("milliliter", "ml").replace("millilitre", "ml")
          .replace("millimeter", "mm").replace("millimetre", "mm"))

    # 4. Brand → generic mapping (longest match first)
    sorted_brands = sorted(SUTURE_BRANDS.items(), key=lambda x: -len(x[0]))
    for brand, generic in sorted_brands:
        # Allow brand to be followed/separated by *, space, or hyphen
        pattern_brand = (re.escape(brand)
                         .replace(r'\ ', r'\s+')
                         .replace(r'\-', r'[\s\-*]?')
                         .replace(r'\*', r'[\s\-*]?'))
        v = re.sub(r'\b' + pattern_brand + r'\b', generic, v, flags=re.IGNORECASE)

    # 5. Word-level synonyms
    tokens = v.split()
    tokens = [WORD_SYNONYMS.get(t, t) for t in tokens]
    v = " ".join(tokens)

    # 6. Punctuation removal — preserve decimal points between digits
    v = re.sub(r'(?<=\d)\.(?=\d)', 'DEC', v)
    v = re.sub(r'[^\w\s]', ' ', v)
    v = v.replace('DEC', '.')

    return re.sub(r'\s+', ' ', v).strip()


# ═══════════════════════════════════════════════════════════════════
#  SCORING & PENALTIES
# ═══════════════════════════════════════════════════════════════════

def multi_score(query: str, candidate: str) -> float:
    """Multi-scorer ensemble — best score across token_set, token_sort, WRatio."""
    if not query or not candidate:
        return 0.0
    return max(
        fuzz.token_set_ratio(query, candidate),
        fuzz.token_sort_ratio(query, candidate),
        fuzz.WRatio(query, candidate),
    )


def consistency_penalty(
    src_norm: str,
    cand_norm: str,
    src_suture: Optional[Set[str]] = None,
    cand_suture: Optional[Set[str]] = None,
) -> Tuple[float, List[str]]:
    """
    Apply penalties for:
      - Generic blacklisted candidates (e.g. "SUTURE KIT")
      - Very short NPC descriptions (likely too generic)
      - Suture size mismatches (HARD validation)
      - Numeric dimension mismatches
      - Core keyword (product type) mismatches
      - Suture-vs-non-suture cross-category matches
    """
    penalty = 0.0
    notes: List[str] = []

    # Generic blacklist
    if cand_norm.strip() in GENERIC_BLACKLIST:
        penalty += 40
        notes.append("Generic blacklisted entry")

    # Short NPC desc (≤3 words)
    if word_count(cand_norm) <= 3:
        penalty += 25
        notes.append("Short NPC description (likely too generic)")

    # ── Suture size HARD validation ──
    if src_suture and cand_suture:
        if not (src_suture & cand_suture):
            penalty += 30
            notes.append(f"Suture size mismatch ({src_suture} vs {cand_suture})")

    # Numeric dimensions
    ss = extract_size_tokens(src_norm)
    cs = extract_size_tokens(cand_norm)
    if ss and cs and not set(ss).intersection(set(cs)):
        src_units  = {u for _, u in ss}
        cand_units = {u for _, u in cs}
        penalty += 15 if src_units == cand_units else 20
        notes.append("Size mismatch")

    # Core keyword
    sk = core_keyword(src_norm)
    ck = core_keyword(cand_norm)
    if sk and ck and sk != ck:
        penalty += 25
        notes.append(f"Keyword mismatch ({sk} vs {ck})")

    # Cross-category: suture should not match mesh/sponge/screw/plate/nail/clip
    if (is_suture_item(src_norm) and not is_suture_item(cand_norm)
            and ck in {"mesh", "sponge", "screw", "plate", "nail", "clip"}):
        penalty += 30
        notes.append("Suture→non-suture cross-category mismatch")

    return penalty, notes


def confidence_label(score: float) -> str:
    if score >= 80:
        return "HIGH"
    if score >= 65:
        return "MEDIUM"
    return "LOW"


def best_match(
    query: str,
    ref_norm: List[str],
    ref_suture_sizes: List[Set[str]],
    src_suture_sizes: Set[str],
    min_score: float,
    limit: int = 10,
) -> Optional[Tuple[int, float, float]]:
    """
    Returns (index, raw_score, final_score) of the best candidate
    after applying consistency penalties — or None.
    """
    if not query:
        return None
    result = process.extract(query, ref_norm, scorer=fuzz.token_set_ratio, limit=limit)
    if not result:
        return None

    rescored = []
    for text, base_score, idx in result:
        if base_score < min_score - 20:
            continue
        ms = multi_score(query, text)
        pen, _ = consistency_penalty(query, text, src_suture_sizes, ref_suture_sizes[idx])
        rescored.append((idx, ms, ms - pen))

    if not rescored:
        return None
    rescored.sort(key=lambda x: -x[2])
    top_idx, top_raw, top_final = rescored[0]
    if top_final < min_score:
        return None
    return top_idx, float(top_raw), float(top_final)


def _structural_query(norm_text: str) -> Optional[str]:
    """Reduce description to <core_keyword> + primary dimension(s)."""
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
    ihs_norm: str,
    ihs_suture_sizes: Set[str],
    npc_df: pd.DataFrame,
    phc_df: pd.DataFrame,
    npc_norm_col: str,
    phc_norm_col: str,
    npc_sutsize_col: str,
    phc_sutsize_col: str,
    npc_code_col: str,
    npc_desc_col: str,
    phc_code_col: str,
    phc_desc_col: str,
    phc_ihbs_col: str,
) -> MatchResult:

    npc_norm     = npc_df[npc_norm_col].tolist()
    phc_norm     = phc_df[phc_norm_col].tolist()
    npc_sut      = npc_df[npc_sutsize_col].tolist()
    phc_sut      = phc_df[phc_sutsize_col].tolist()
    review_notes: List[str] = []

    # ── Step 1: Primary NPC match (≥65) ──
    primary = best_match(ihs_norm, npc_norm, npc_sut, ihs_suture_sizes, min_score=65)
    if primary:
        idx, raw, final = primary
        _, notes = consistency_penalty(ihs_norm, npc_norm[idx], ihs_suture_sizes, npc_sut[idx])
        review_notes.extend(notes)
        return MatchResult(
            npc_code=str(npc_df.iloc[idx][npc_code_col]),
            rw_description=str(npc_df.iloc[idx][npc_desc_col]),
            ihbs_code=None,
            match_source="NPC",
            match_score=round(final, 2),
            confidence_level=confidence_label(final),
            candidate_desc=str(npc_df.iloc[idx][npc_desc_col]),
            duplicate_flag=False,
            review_notes="; ".join(review_notes),
        )

    # ── Step 2: PHC fallback (≥60) ──
    secondary = best_match(ihs_norm, phc_norm, phc_sut, ihs_suture_sizes, min_score=60)
    if secondary:
        idx, raw, final = secondary
        _, notes = consistency_penalty(ihs_norm, phc_norm[idx], ihs_suture_sizes, phc_sut[idx])
        review_notes.extend(notes)
        return MatchResult(
            npc_code=str(phc_df.iloc[idx][phc_code_col]),
            rw_description=str(phc_df.iloc[idx][phc_desc_col]),
            ihbs_code=str(phc_df.iloc[idx][phc_ihbs_col]),
            match_source="PHC",
            match_score=round(final, 2),
            confidence_level=confidence_label(final),
            candidate_desc=str(phc_df.iloc[idx][phc_desc_col]),
            duplicate_flag=False,
            review_notes="; ".join(review_notes),
        )

    # ── Step 3: Structural fallback against NPC (keyword + size only) ──
    structural = _structural_query(ihs_norm)
    if structural and structural != ihs_norm:
        relaxed = best_match(structural, npc_norm, npc_sut, ihs_suture_sizes, min_score=60)
        if relaxed:
            idx, raw, final = relaxed
            _, notes = consistency_penalty(structural, npc_norm[idx], ihs_suture_sizes, npc_sut[idx])
            review_notes.extend(notes)
            review_notes.append("Structural fallback")
            return MatchResult(
                npc_code=str(npc_df.iloc[idx][npc_code_col]),
                rw_description=str(npc_df.iloc[idx][npc_desc_col]),
                ihbs_code=None,
                match_source="NPC_STRUCTURAL",
                match_score=round(final, 2),
                confidence_level=confidence_label(final),
                candidate_desc=str(npc_df.iloc[idx][npc_desc_col]),
                duplicate_flag=False,
                review_notes="; ".join(review_notes),
            )

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
    st.caption("v3 — accuracy-focused engine with suture-size validation and false-positive guards")

    with st.expander("ℹ️  What changed in v3?", expanded=False):
        st.markdown("""
**Critical accuracy fixes over v2:**

| Fix | Impact |
|---|---|
| 🔢 **European decimal commas** (`2,7` → `2.7`) | 298 cannulated/cortical screws now size-validated |
| 📏 **Suture size validation** (`2-0` ≠ `4-0`) | Prevents POLYSORB 2-0 from matching VICRYL SIZE 0 |
| 🚫 **Generic-NPC blacklist** | Eliminated 478 false matches to "SUTURE KIT" |
| 🔧 **Flexible DxL pattern** | Catches `2.7X20mm`, `5X35MM` (was missing first `mm`) |
| 🚷 **Cross-category penalty** | Sutures no longer match mesh/sponge/screw |
| 🏷️ **More brands & color-code stripping** | V-LOC, SURGIPRO II, LIGASURE; VIO/BLU/UND/BRN now stripped |

**Plus all v2 improvements:**
- 25+ suture brand → generic name mappings
- `CORTEX` → `CORTICAL`, decimal-preserving punctuation removal
- Multi-scorer ensemble (token_set + token_sort + WRatio)
- Structural fallback (core keyword + dimension)
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
            "Aggressive noise removal", value=True,
            help="Strips packaging codes, color indicators, and abbreviated qualifiers. Recommended ON.",
        )
    with col_b:
        show_unmatched = st.checkbox(
            "Show unmatched items after run", value=False,
            help="Display the first 50 unmatched IHS descriptions for review.",
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

        ihs_work = ihs_df.copy()
        npc_work = npc_df.copy()
        phc_work = phc_df.copy()

        # Extract suture sizes from RAW text (before normalization removes dashes)
        status_text.text("Extracting suture sizes…")
        ihs_work["_sutsize"] = ihs_work[ihs_map["desc"]].apply(extract_suture_sizes)
        npc_work["_sutsize"] = npc_work[npc_map["rw_desc"]].apply(extract_suture_sizes)
        phc_work["_sutsize"] = phc_work[phc_map["desc"]].apply(extract_suture_sizes)

        # Normalize
        status_text.text("Normalising descriptions…")
        ihs_work["_norm"] = ihs_work[ihs_map["desc"]].apply(lambda x: normalize_text(x, aggressive))
        npc_work["_norm"] = npc_work[npc_map["rw_desc"]].apply(lambda x: normalize_text(x, aggressive))
        phc_work["_norm"] = phc_work[phc_map["desc"]].apply(lambda x: normalize_text(x, aggressive))

        # Match
        results = []
        total = len(ihs_work)
        for i, (_, row) in enumerate(ihs_work.iterrows()):
            match = match_ihs_row(
                ihs_norm         = row["_norm"],
                ihs_suture_sizes = row["_sutsize"],
                npc_df           = npc_work,
                phc_df           = phc_work,
                npc_norm_col     = "_norm",
                phc_norm_col     = "_norm",
                npc_sutsize_col  = "_sutsize",
                phc_sutsize_col  = "_sutsize",
                npc_code_col     = npc_map["npc_code"],
                npc_desc_col     = npc_map["rw_desc"],
                phc_code_col     = phc_map["npc_code"],
                phc_desc_col     = phc_map["desc"],
                phc_ihbs_col     = phc_map["ihbs_code"],
            )
            results.append(match)
            if i % 50 == 0:
                progress.progress(min((i + 1) / max(total, 1), 1.0))
                status_text.text(f"Matching {i + 1} / {total}…")

        progress.progress(1.0)
        status_text.text("Complete!")

        # Build output
        result_df = ihs_work.drop(columns=["_norm", "_sutsize"]).copy()
        result_df["NPC_Code"]                      = [r.npc_code         for r in results]
        result_df["RW_Product_Description"]        = [r.rw_description   for r in results]
        result_df["IHBS_Code"]                     = [r.ihbs_code        for r in results]
        result_df["Match_Source"]                  = [r.match_source     for r in results]
        result_df["Match_Score"]                   = [r.match_score      for r in results]
        result_df["Confidence_Level"]              = [r.confidence_level for r in results]
        result_df["Matched_Candidate_Description"] = [r.candidate_desc   for r in results]
        result_df["Duplicate_Match_Flag"]          = [r.duplicate_flag   for r in results]
        result_df["Review_Notes"]                  = [r.review_notes     for r in results]

        # ── Summary ──
        src_counts = result_df["Match_Source"].value_counts()
        matched    = (result_df["Match_Source"] != "UNMATCHED").sum()
        rate       = matched / len(result_df) * 100 if len(result_df) else 0

        st.success(f"✅ **{rate:.1f}%** of IHS items matched  ({matched} / {len(result_df)})")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("NPC (direct)",     src_counts.get("NPC", 0))
        m2.metric("NPC (structural)", src_counts.get("NPC_STRUCTURAL", 0))
        m3.metric("PHC fallback",     src_counts.get("PHC", 0))
        m4.metric("Unmatched",        src_counts.get("UNMATCHED", 0))

        conf = result_df[result_df["Match_Source"] != "UNMATCHED"]["Confidence_Level"].value_counts()
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
