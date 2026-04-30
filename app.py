"""
Consumables Master Data Harmonization Tool  (IHS ↔ NPC ↔ PHC)
v4 — enhanced fuzzy matching + verification workspace

NEW IN v4:
  • Medical abbreviation expansion (CANN→cannulated, DCP→dynamic compression plate, ...)
  • Jaro-Winkler similarity added to multi-scorer ensemble
  • TF-IDF specificity bonus (rewards matches on rare/specific tokens)
  • Improved suture-size extraction (catches bare-digit sizes after brand names)
  • VERIFICATION TAB: review, accept, reject, override and re-export matches
"""
import io
import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, process, distance

st.set_page_config(page_title="Consumables Master Data Harmonization", layout="wide")

# ═══════════════════════════════════════════════════════════════════
#  DICTIONARIES
# ═══════════════════════════════════════════════════════════════════

SUTURE_BRANDS: Dict[str, str] = {
    # Multi-word phrases first (longest match wins)
    "surgipro ii":   "polypropylene suture",
    "surgipro*ii":   "polypropylene suture",
    "surgiproii":    "polypropylene suture",
    "v loc":         "barbed absorbable suture",
    "v-loc":         "barbed absorbable suture",
    "vloc":          "barbed absorbable suture",
    "hem-o-lok":     "hem o lok ligation clip",
    "hem o lok":     "hem o lok ligation clip",
    "hemolock":      "hem o lok ligation clip",
    # Suture brands
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
    # Surgical devices
    "ligasure":      "vessel sealing device",
    "versaport":     "laparoscopic trocar port",
    "autosonix":     "ultrasonic dissector",
    # Generic
    "injector":      "syringe",
    "hand gloves":   "gloves",
    "iv line":       "cannula",
}

WORD_SYNONYMS: Dict[str, str] = {
    "cortex":      "cortical",
    "concelous":   "cancellous",
    "cancellus":   "cancellous",
    "concelouos":  "cancellous",
    "milliliter":  "ml",  "millilitre":  "ml",
    "millimeter":  "mm",  "millimetre":  "mm",
}

# NEW: medical abbreviations expanded after tokenisation
MEDICAL_ABBREVIATIONS: Dict[str, str] = {
    "fg": "french", "fr": "french", "ch": "french",
    "rb": "round body", "rc": "reverse cutting",
    "tp": "taper point", "tc": "taper cut", "cir": "circle",
    "chrm": "chromic", "chrome": "chromic",
    "absorb": "absorbable", "absorbab": "absorbable",
    "monoflt": "monofilament", "multiflt": "multifilament",
    "reusbl": "reusable", "dispbl": "disposable",
    "std": "standard", "sm": "small", "lg": "large",
    "lcp": "locking compression plate",
    "dhs": "dynamic hip screw",
    "dcp": "dynamic compression plate",
    "tens": "titanium elastic nail",
    "kw": "kirschner wire", "ks": "kirschner",
    "th": "threaded", "thr": "threaded", "thrd": "threaded",
    "cann": "cannulated", "canc": "cancellous", "cort": "cortical",
    "lkg": "locking",
    "prox": "proximal", "dist": "distal", "lat": "lateral",
    "titan": "titanium", "ti": "titanium", "ss": "stainless steel",
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

GENERIC_BLACKLIST = {
    "suture kit", "endo z bur", "iris forceps", "sealing machine",
    "reusable apron", "d plate", "filter 1µ", "mixing spatula",
    "ise control i", "magill forcep", "skin stapler", "bipolar forceps cord",
}

NOISE_PATTERNS: List[re.Pattern] = [
    re.compile(r'\(pitch[^)]*\)',           re.IGNORECASE),
    re.compile(r'\bpitch\s*[\d.,]*',        re.IGNORECASE),
    re.compile(r'\bself[-\s]*tapp?ing\b',   re.IGNORECASE),
    re.compile(r'\b[sS]\.[tT]\.?\b'),
    re.compile(r'\b(vio|blu|brn|blk|und|whi|dlu)\b', re.IGNORECASE),
    re.compile(r'\bx\s*\d+\b(?!\s*mm)(?!\s*cm)', re.IGNORECASE),
    re.compile(r'\b\d+\s*x\s*\d+\s*cm\b',   re.IGNORECASE),
    re.compile(r'\b(dt|da)\b',              re.IGNORECASE),
    re.compile(r'\b(gs|cv|uc|mc|mv|btp|bb|kv)\d+\b', re.IGNORECASE),
    re.compile(r'\b[a-z]{2}\d{2,4}\b',      re.IGNORECASE),
    re.compile(r'\b(premium|disp[oo]?|assy|sterile|uu|estch)\b', re.IGNORECASE),
    re.compile(r'\blp\b(?!\d)',             re.IGNORECASE),
    re.compile(r'\bsingle\s*use\b',         re.IGNORECASE),
    re.compile(r'\bligatures?\b',           re.IGNORECASE),
    re.compile(r'\blig\b',                  re.IGNORECASE),
    re.compile(r'\bnon\s*abs\b',            re.IGNORECASE),
    re.compile(r'\(d\)',                    re.IGNORECASE),
    re.compile(r'\*'),
]

DIM_X_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:mm)?\s*x\s*(\d+(?:\.\d+)?)\s*mm', re.IGNORECASE
)
UNIT_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(mm|ml|cc|fr|fg|ch|g|mg|mcg)\b', re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════
#  EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════════

def extract_suture_sizes(raw_text: object) -> Set[str]:
    """
    Extract suture size codes from RAW text (before normalization).
    Catches:
      - n-0 / n/0 patterns (e.g. '2-0', '4/0', '8-0BLU')
      - 'SIZE n' / 'SIZE n-0'
      - Bare digit AFTER suture brand or material name
    """
    if pd.isna(raw_text):
        return set()
    text = str(raw_text)
    sizes: Set[str] = set()

    # n-0 / n/0
    for m in re.findall(r'\b(\d{1,2})\s*[-/]\s*0(?=\D|$)', text):
        sizes.add(f"{m}-0")

    # SIZE n
    for m in re.finditer(r'\bsize\s*:?\s*(\d{1,2})(?:\s*[-/]\s*0)?', text, re.IGNORECASE):
        full = m.group(0).lower()
        n = m.group(1)
        sizes.add(f"{n}-0" if ("-0" in full or "/0" in full) else n)

    # bare digit after suture brand/material name
    suture_terms = (r"polysorb|polycryl|velosorb|biosyn|caprosyn|surgipro|ticron|"
                    r"v[\s\-*]?loc|monosof|sofsilk|silkam|vicryl|pds|prolene|ethilon|"
                    r"polyglactin|polyglecaprone|polypropylene|polydioxanone|nylon|"
                    r"silk|catgut|chromic|maxon|mersilene|ethibond|novafil")
    pat = (rf"\b({suture_terms})\b[\s*\-]*(?:\(vicryl\))?[\s*\-]*(?:ii)?[\s*\-]*"
           rf"(?:910|6211|631|25)?\s*[\s*\-]+(\d{{1,2}})\b(?!\s*mm)(?!\s*cm)")
    for m in re.finditer(pat, text, re.IGNORECASE):
        n = m.group(2)
        end = m.end(2)
        rest = text[end:end + 5]
        sizes.add(f"{n}-0" if re.match(r'\s*[-/]\s*0', rest) else n)

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


def expand_abbreviations(text: str) -> str:
    return " ".join(MEDICAL_ABBREVIATIONS.get(t, t) for t in text.split())


# ═══════════════════════════════════════════════════════════════════
#  NORMALIZATION
# ═══════════════════════════════════════════════════════════════════

def normalize_text(text: object, aggressive: bool = True) -> str:
    if pd.isna(text):
        return ""
    v = str(text).lower().strip()

    # 0. European decimal commas: "2,7" → "2.7"
    v = re.sub(r'(?<=\d),(?=\d)', '.', v)

    # 1. Flexible DxL dimension expansion
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
        pattern_brand = (re.escape(brand)
                         .replace(r'\ ', r'\s+')
                         .replace(r'\-', r'[\s\-*]?')
                         .replace(r'\*', r'[\s\-*]?'))
        v = re.sub(r'\b' + pattern_brand + r'\b', generic, v, flags=re.IGNORECASE)

    # 5. Word-level synonyms
    tokens = v.split()
    tokens = [WORD_SYNONYMS.get(t, t) for t in tokens]
    v = " ".join(tokens)

    # 6. NEW: medical abbreviation expansion
    v = expand_abbreviations(v)

    # 7. Punctuation removal — preserve decimal points between digits
    v = re.sub(r'(?<=\d)\.(?=\d)', 'DEC', v)
    v = re.sub(r'[^\w\s]', ' ', v)
    v = v.replace('DEC', '.')

    return re.sub(r'\s+', ' ', v).strip()


# ═══════════════════════════════════════════════════════════════════
#  FUZZY MATCHING ENGINE (enhanced)
# ═══════════════════════════════════════════════════════════════════

# IDF cache — populated when corpus is loaded
_IDF_CACHE: Dict[str, float] = {}


def compute_idf(all_descriptions: List[str]) -> Dict[str, float]:
    """Compute inverse document frequency for each token in the combined corpus."""
    df_count: Counter = Counter()
    for desc in all_descriptions:
        for tok in set(desc.split()):
            df_count[tok] += 1
    n = len(all_descriptions)
    return {t: math.log(n / max(c, 1)) for t, c in df_count.items()}


def specificity_bonus(query: str, candidate: str) -> float:
    """0–10 bonus when matched tokens are rare/specific (high IDF)."""
    if not _IDF_CACHE:
        return 0.0
    q = set(query.split())
    c = set(candidate.split())
    inter = q & c
    if not inter:
        return 0.0
    avg_idf = sum(_IDF_CACHE.get(t, 1.0) for t in inter) / len(inter)
    if avg_idf < 3:
        return 0.0
    if avg_idf > 5:
        return 10.0
    return (avg_idf - 3) / 2 * 10


def jaro_winkler_score(q: str, c: str) -> float:
    if not q or not c:
        return 0.0
    return distance.JaroWinkler.normalized_similarity(q, c) * 100


def multi_score(query: str, candidate: str) -> float:
    """
    Multi-scorer ensemble: best of token_set + token_sort + WRatio + Jaro-Winkler,
    plus a specificity bonus when matched tokens are rare/specific.
    """
    if not query or not candidate:
        return 0.0
    base = max(
        fuzz.token_set_ratio(query, candidate),
        fuzz.token_sort_ratio(query, candidate),
        fuzz.WRatio(query, candidate),
        jaro_winkler_score(query, candidate),
    )
    bonus = specificity_bonus(query, candidate) if base >= 60 else 0.0
    return min(100.0, base + bonus)


def consistency_penalty(
    src_norm: str,
    cand_norm: str,
    src_suture: Optional[Set[str]] = None,
    cand_suture: Optional[Set[str]] = None,
) -> Tuple[float, List[str]]:
    penalty = 0.0
    notes: List[str] = []

    if cand_norm.strip() in GENERIC_BLACKLIST:
        penalty += 40
        notes.append("Generic blacklisted entry")

    if word_count(cand_norm) <= 3:
        penalty += 25
        notes.append("Short NPC description")

    if src_suture and cand_suture:
        if not (src_suture & cand_suture):
            penalty += 30
            notes.append(f"Suture size mismatch ({src_suture} vs {cand_suture})")

    ss = extract_size_tokens(src_norm)
    cs = extract_size_tokens(cand_norm)
    if ss and cs:
        src_set = set(ss)
        cand_set = set(cs)
        if not src_set.intersection(cand_set):
            src_units  = {u for _, u in ss}
            cand_units = {u for _, u in cs}
            penalty += 15 if src_units == cand_units else 20
            notes.append("Size mismatch (no dimension overlap)")
        else:
            # Some overlap — but check if any source dimension is missing from candidate
            missing = src_set - cand_set
            if missing:
                penalty += 8 * len(missing)  # ~8 per missing dimension
                notes.append(f"Partial size mismatch (missing {missing})")

    sk = core_keyword(src_norm)
    ck = core_keyword(cand_norm)
    if sk and ck and sk != ck:
        penalty += 25
        notes.append(f"Keyword mismatch ({sk} vs {ck})")

    if (is_suture_item(src_norm) and not is_suture_item(cand_norm)
            and ck in {"mesh", "sponge", "screw", "plate", "nail", "clip"}):
        penalty += 30
        notes.append("Suture→non-suture mismatch")

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
) -> Optional[Tuple[int, float, float, List[str]]]:
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
        pen, notes = consistency_penalty(query, text, src_suture_sizes, ref_suture_sizes[idx])
        rescored.append((idx, ms, ms - pen, notes))

    if not rescored:
        return None
    rescored.sort(key=lambda x: -x[2])
    top_idx, top_raw, top_final, notes = rescored[0]
    if top_final < min_score:
        return None
    return top_idx, float(top_raw), float(top_final), notes


def _structural_query(norm_text: str) -> Optional[str]:
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

@dataclass
class MatchResult:
    npc_code: Optional[str] = None
    rw_description: Optional[str] = None
    ihbs_code: Optional[str] = None
    match_source: str = "UNMATCHED"
    match_score: float = 0.0
    confidence_level: str = "LOW"
    candidate_desc: Optional[str] = None
    duplicate_flag: bool = False
    review_notes: str = "No suitable match found"


def match_ihs_row(
    ihs_norm: str, ihs_suture_sizes: Set[str],
    npc_df: pd.DataFrame, phc_df: pd.DataFrame,
    npc_norm_col: str, phc_norm_col: str,
    npc_sutsize_col: str, phc_sutsize_col: str,
    npc_code_col: str, npc_desc_col: str,
    phc_code_col: str, phc_desc_col: str, phc_ihbs_col: str,
) -> MatchResult:

    npc_norm = npc_df[npc_norm_col].tolist()
    phc_norm = phc_df[phc_norm_col].tolist()
    npc_sut  = npc_df[npc_sutsize_col].tolist()
    phc_sut  = phc_df[phc_sutsize_col].tolist()

    # Step 1: NPC primary (≥65)
    primary = best_match(ihs_norm, npc_norm, npc_sut, ihs_suture_sizes, 65)
    if primary:
        idx, raw, final, notes = primary
        return MatchResult(
            npc_code=str(npc_df.iloc[idx][npc_code_col]),
            rw_description=str(npc_df.iloc[idx][npc_desc_col]),
            match_source="NPC",
            match_score=round(final, 2),
            confidence_level=confidence_label(final),
            candidate_desc=str(npc_df.iloc[idx][npc_desc_col]),
            review_notes="; ".join(notes),
        )

    # Step 2: PHC fallback (≥60)
    secondary = best_match(ihs_norm, phc_norm, phc_sut, ihs_suture_sizes, 60)
    if secondary:
        idx, raw, final, notes = secondary
        return MatchResult(
            npc_code=str(phc_df.iloc[idx][phc_code_col]),
            rw_description=str(phc_df.iloc[idx][phc_desc_col]),
            ihbs_code=str(phc_df.iloc[idx][phc_ihbs_col]),
            match_source="PHC",
            match_score=round(final, 2),
            confidence_level=confidence_label(final),
            candidate_desc=str(phc_df.iloc[idx][phc_desc_col]),
            review_notes="; ".join(notes),
        )

    # Step 3: structural fallback
    structural = _structural_query(ihs_norm)
    if structural and structural != ihs_norm:
        relaxed = best_match(structural, npc_norm, npc_sut, ihs_suture_sizes, 60)
        if relaxed:
            idx, raw, final, notes = relaxed
            notes.append("Structural fallback")
            return MatchResult(
                npc_code=str(npc_df.iloc[idx][npc_code_col]),
                rw_description=str(npc_df.iloc[idx][npc_desc_col]),
                match_source="NPC_STRUCTURAL",
                match_score=round(final, 2),
                confidence_level=confidence_label(final),
                candidate_desc=str(npc_df.iloc[idx][npc_desc_col]),
                review_notes="; ".join(notes),
            )

    return MatchResult()


# ═══════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════

def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "IHS_Enriched") -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
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


def lookup_npc_description(npc_code: str, npc_df: pd.DataFrame,
                            code_col: str, desc_col: str) -> Optional[str]:
    """Look up NPC description by code (used in verification override)."""
    if not npc_code or pd.isna(npc_code):
        return None
    matches = npc_df[npc_df[code_col].astype(str) == str(npc_code)]
    if not matches.empty:
        return str(matches.iloc[0][desc_col])
    return None


# ═══════════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize persistent state across reruns."""
    defaults = {
        "results_df":      None,
        "npc_df_cached":   None,
        "phc_df_cached":   None,
        "ihs_col_name":    None,
        "npc_code_col":    None,
        "npc_desc_col":    None,
        "verified_df":     None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ─── TAB 1 ────────────────────────────────────────────────────────
def render_setup_and_run_tab():
    st.markdown("### 1️⃣  Upload datasets")
    c1, c2, c3 = st.columns(3)
    with c1:
        ihs_file = st.file_uploader("IHS File", type=["xlsx", "xls"], key="ihs_file")
    with c2:
        npc_file = st.file_uploader("NPC File", type=["xlsx", "xls"], key="npc_file")
    with c3:
        phc_file = st.file_uploader("PHC File", type=["xlsx", "xls"], key="phc_file")

    if not (ihs_file and npc_file and phc_file):
        st.info("Upload all three Excel files to continue.")
        return

    ihs_df = pd.read_excel(ihs_file)
    npc_df = pd.read_excel(npc_file)
    phc_df = pd.read_excel(phc_file)

    with st.expander("Dataset preview", expanded=False):
        tab1, tab2, tab3 = st.tabs(["IHS", "NPC", "PHC"])
        with tab1: st.dataframe(ihs_df.head(10), use_container_width=True)
        with tab2: st.dataframe(npc_df.head(10), use_container_width=True)
        with tab3: st.dataframe(phc_df.head(10), use_container_width=True)

    st.markdown("### 2️⃣  Map columns")
    ihs_map = mapping_ui(ihs_df, "IHS", {"Product Name / Description": "desc"})
    npc_map = mapping_ui(npc_df, "NPC",
        {"NPC Code": "npc_code", "RW Product Description": "rw_desc"})
    phc_map = mapping_ui(phc_df, "PHC",
        {"Product Description": "desc", "NPC Code": "npc_code", "IHBS Code": "ihbs_code"})

    st.markdown("### 3️⃣  Options")
    col_a, col_b = st.columns(2)
    with col_a:
        aggressive = st.checkbox("Aggressive noise removal", value=True)
    with col_b:
        use_specificity = st.checkbox(
            "Use TF-IDF specificity bonus", value=True,
            help="Reward matches that share rare/informative tokens (recommended).",
        )

    st.markdown("### 4️⃣  Run matching")
    if not st.button("▶  Run Matching", type="primary"):
        return

    all_maps = [ihs_map["desc"], npc_map["npc_code"], npc_map["rw_desc"],
                phc_map["desc"], phc_map["npc_code"], phc_map["ihbs_code"]]
    if any(not m for m in all_maps):
        st.error("Please map all required columns before running matching.")
        return

    progress = st.progress(0)
    status = st.empty()

    ihs_work = ihs_df.copy()
    npc_work = npc_df.copy()
    phc_work = phc_df.copy()

    status.text("Extracting suture sizes…")
    ihs_work["_sutsize"] = ihs_work[ihs_map["desc"]].apply(extract_suture_sizes)
    npc_work["_sutsize"] = npc_work[npc_map["rw_desc"]].apply(extract_suture_sizes)
    phc_work["_sutsize"] = phc_work[phc_map["desc"]].apply(extract_suture_sizes)

    status.text("Normalising descriptions…")
    ihs_work["_norm"] = ihs_work[ihs_map["desc"]].apply(lambda x: normalize_text(x, aggressive))
    npc_work["_norm"] = npc_work[npc_map["rw_desc"]].apply(lambda x: normalize_text(x, aggressive))
    phc_work["_norm"] = phc_work[phc_map["desc"]].apply(lambda x: normalize_text(x, aggressive))

    if use_specificity:
        status.text("Computing TF-IDF weights…")
        global _IDF_CACHE
        all_desc = ihs_work["_norm"].tolist() + npc_work["_norm"].tolist() + phc_work["_norm"].tolist()
        _IDF_CACHE = compute_idf(all_desc)
    else:
        _IDF_CACHE = {}

    results = []
    total = len(ihs_work)
    for i, (_, row) in enumerate(ihs_work.iterrows()):
        m = match_ihs_row(
            ihs_norm=row["_norm"], ihs_suture_sizes=row["_sutsize"],
            npc_df=npc_work, phc_df=phc_work,
            npc_norm_col="_norm", phc_norm_col="_norm",
            npc_sutsize_col="_sutsize", phc_sutsize_col="_sutsize",
            npc_code_col=npc_map["npc_code"], npc_desc_col=npc_map["rw_desc"],
            phc_code_col=phc_map["npc_code"], phc_desc_col=phc_map["desc"],
            phc_ihbs_col=phc_map["ihbs_code"],
        )
        results.append(m)
        if i % 50 == 0:
            progress.progress(min((i + 1) / max(total, 1), 1.0))
            status.text(f"Matching {i + 1} / {total}…")

    progress.progress(1.0)
    status.text("Complete!")

    # Build result dataframe
    out = ihs_work.drop(columns=["_norm", "_sutsize"]).copy()
    out["NPC_Code"]                      = [r.npc_code for r in results]
    out["RW_Product_Description"]        = [r.rw_description for r in results]
    out["IHBS_Code"]                     = [r.ihbs_code for r in results]
    out["Match_Source"]                  = [r.match_source for r in results]
    out["Match_Score"]                   = [r.match_score for r in results]
    out["Confidence_Level"]              = [r.confidence_level for r in results]
    out["Matched_Candidate_Description"] = [r.candidate_desc for r in results]
    out["Duplicate_Match_Flag"]          = [r.duplicate_flag for r in results]
    out["Review_Notes"]                  = [r.review_notes for r in results]
    # Verification columns (initially blank)
    out["Decision"]        = "Pending"   # Pending / Accept / Reject / Modify
    out["Verified_NPC_Code"]    = ""
    out["Verified_Description"] = ""
    out["Reviewer_Notes"]  = ""

    # Persist to session_state
    st.session_state.results_df    = out
    st.session_state.npc_df_cached = npc_df
    st.session_state.phc_df_cached = phc_df
    st.session_state.ihs_col_name  = ihs_map["desc"]
    st.session_state.npc_code_col  = npc_map["npc_code"]
    st.session_state.npc_desc_col  = npc_map["rw_desc"]

    # ── Summary ──
    src = out["Match_Source"].value_counts()
    matched = (out["Match_Source"] != "UNMATCHED").sum()
    rate = matched / len(out) * 100 if len(out) else 0
    st.success(f"✅ **{rate:.1f}%** matched  ({matched} / {len(out)})  ·  Move to **Verify Matches** tab to review.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("NPC (direct)", src.get("NPC", 0))
    m2.metric("NPC (structural)", src.get("NPC_STRUCTURAL", 0))
    m3.metric("PHC fallback", src.get("PHC", 0))
    m4.metric("Unmatched", src.get("UNMATCHED", 0))

    conf = out[out["Match_Source"] != "UNMATCHED"]["Confidence_Level"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("HIGH",   conf.get("HIGH", 0))
    c2.metric("MEDIUM", conf.get("MEDIUM", 0))
    c3.metric("LOW",    conf.get("LOW", 0))

    st.markdown("### Output Preview")
    st.dataframe(out.head(50), use_container_width=True)


# ─── TAB 2 ────────────────────────────────────────────────────────
def render_verification_tab():
    if st.session_state.results_df is None:
        st.info("👈  Run matching first in the **Setup & Run** tab.")
        return

    df: pd.DataFrame = st.session_state.results_df
    ihs_col      = st.session_state.ihs_col_name
    npc_df       = st.session_state.npc_df_cached
    npc_code_col = st.session_state.npc_code_col
    npc_desc_col = st.session_state.npc_desc_col

    st.markdown("Use this workspace to review each matched item, override incorrect "
                "matches, and approve the final mapping.")

    # ── Filters ──
    st.markdown("#### 🔍 Filter rows for review")
    f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
    with f1:
        conf_filter = st.multiselect(
            "Confidence", options=["HIGH", "MEDIUM", "LOW"], default=["MEDIUM", "LOW"],
            help="MEDIUM and LOW most need human review.",
        )
    with f2:
        source_filter = st.multiselect(
            "Match source", options=df["Match_Source"].unique().tolist(),
            default=df["Match_Source"].unique().tolist(),
        )
    with f3:
        decision_filter = st.multiselect(
            "Decision status", options=["Pending", "Accept", "Reject", "Modify"],
            default=["Pending"],
        )
    with f4:
        search = st.text_input("Search (IHS description)", "").strip()

    mask = (df["Confidence_Level"].isin(conf_filter)
            & df["Match_Source"].isin(source_filter)
            & df["Decision"].isin(decision_filter))
    if search:
        mask &= df[ihs_col].astype(str).str.contains(search, case=False, na=False)

    filtered = df[mask].copy()
    st.caption(f"Showing **{len(filtered)}** of {len(df)} rows")

    # ── Bulk actions ──
    st.markdown("#### ⚡ Bulk actions on the filtered rows")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("✅ Accept all filtered", use_container_width=True):
            df.loc[mask, "Decision"] = "Accept"
            st.session_state.results_df = df
            st.rerun()
    with b2:
        if st.button("❌ Reject all filtered", use_container_width=True):
            df.loc[mask, "Decision"] = "Reject"
            st.session_state.results_df = df
            st.rerun()
    with b3:
        if st.button("↩️  Reset all filtered to Pending", use_container_width=True):
            df.loc[mask, "Decision"] = "Pending"
            df.loc[mask, "Verified_NPC_Code"] = ""
            df.loc[mask, "Verified_Description"] = ""
            st.session_state.results_df = df
            st.rerun()
    with b4:
        if st.button("✨ Auto-accept HIGH confidence (whole dataset)", use_container_width=True):
            df.loc[(df["Confidence_Level"] == "HIGH") & (df["Decision"] == "Pending"),
                   "Decision"] = "Accept"
            st.session_state.results_df = df
            st.rerun()

    # ── Editable verification table ──
    st.markdown("#### 📋 Review & edit (changes save automatically)")
    display_cols = [
        ihs_col, "NPC_Code", "Matched_Candidate_Description",
        "Confidence_Level", "Match_Score", "Review_Notes",
        "Decision", "Verified_NPC_Code", "Reviewer_Notes",
    ]
    edit_df = filtered[display_cols].copy().reset_index(drop=False).rename(columns={"index": "_orig_idx"})

    edited = st.data_editor(
        edit_df,
        column_config={
            "_orig_idx": st.column_config.NumberColumn("Row", disabled=True, width="small"),
            ihs_col: st.column_config.TextColumn("IHS Description", disabled=True, width="large"),
            "NPC_Code": st.column_config.TextColumn("Matched NPC Code", disabled=True, width="medium"),
            "Matched_Candidate_Description": st.column_config.TextColumn("Matched NPC Description", disabled=True, width="large"),
            "Confidence_Level": st.column_config.TextColumn("Conf.", disabled=True, width="small"),
            "Match_Score": st.column_config.NumberColumn("Score", disabled=True, format="%.1f", width="small"),
            "Review_Notes": st.column_config.TextColumn("Notes", disabled=True, width="medium"),
            "Decision": st.column_config.SelectboxColumn(
                "Decision", options=["Pending", "Accept", "Reject", "Modify"], width="small",
            ),
            "Verified_NPC_Code": st.column_config.TextColumn(
                "Override NPC Code", width="medium",
                help="Type the correct NPC code if Decision is 'Modify'.",
            ),
            "Reviewer_Notes": st.column_config.TextColumn("Reviewer Notes", width="medium"),
        },
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        key="verify_editor",
    )

    # Apply edits back into the master dataframe
    if edited is not None:
        for _, row in edited.iterrows():
            orig_idx = int(row["_orig_idx"])
            df.at[orig_idx, "Decision"]          = row["Decision"]
            df.at[orig_idx, "Verified_NPC_Code"] = row["Verified_NPC_Code"]
            df.at[orig_idx, "Reviewer_Notes"]    = row["Reviewer_Notes"]
            # Auto-look-up override description
            if row["Verified_NPC_Code"] and str(row["Verified_NPC_Code"]).strip():
                desc = lookup_npc_description(
                    str(row["Verified_NPC_Code"]).strip(), npc_df, npc_code_col, npc_desc_col,
                )
                df.at[orig_idx, "Verified_Description"] = desc or "(NPC code not found)"
            elif row["Decision"] == "Accept":
                df.at[orig_idx, "Verified_Description"] = df.at[orig_idx, "Matched_Candidate_Description"]
        st.session_state.results_df = df

    # ── NPC code search helper ──
    with st.expander("🔎 Search NPC catalog (find a code to override with)"):
        search_npc = st.text_input("Search NPC descriptions", key="npc_search").strip()
        if search_npc:
            hits = npc_df[npc_df[npc_desc_col].astype(str).str.contains(search_npc, case=False, na=False)]
            st.dataframe(hits[[npc_code_col, npc_desc_col]].head(20), use_container_width=True)
        else:
            st.caption("Enter a search term above to find NPC entries.")


# ─── TAB 3 ────────────────────────────────────────────────────────
def render_export_tab():
    if st.session_state.results_df is None:
        st.info("👈  Run matching first in the **Setup & Run** tab.")
        return

    df: pd.DataFrame = st.session_state.results_df

    # Decision counts
    dec_counts = df["Decision"].value_counts()
    st.markdown("### Verification status")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("✅ Accepted",  dec_counts.get("Accept", 0))
    d2.metric("✏️  Modified",  dec_counts.get("Modify", 0))
    d3.metric("❌ Rejected",  dec_counts.get("Reject", 0))
    d4.metric("⏳ Pending",    dec_counts.get("Pending", 0))

    accepted = dec_counts.get("Accept", 0) + dec_counts.get("Modify", 0)
    pct = accepted / len(df) * 100 if len(df) else 0
    st.progress(pct / 100, text=f"{pct:.1f}% verified ({accepted}/{len(df)})")

    # ── Build the final verified dataframe ──
    final_df = df.copy()
    # If Decision is Accept, copy matched values to Verified columns
    accept_mask = final_df["Decision"] == "Accept"
    final_df.loc[accept_mask, "Verified_NPC_Code"] = final_df.loc[accept_mask, "NPC_Code"].fillna("")
    final_df.loc[accept_mask, "Verified_Description"] = final_df.loc[accept_mask, "Matched_Candidate_Description"].fillna("")
    # If Decision is Reject, blank them
    reject_mask = final_df["Decision"] == "Reject"
    final_df.loc[reject_mask, "Verified_NPC_Code"] = ""
    final_df.loc[reject_mask, "Verified_Description"] = ""

    st.markdown("### Download options")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️  Download FULL output (with auto-match + verification columns)",
            data=to_excel_bytes(final_df, "IHS_Enriched"),
            file_name="IHS_Enriched_With_NPC.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with c2:
        # Verified-only (Accept + Modify) clean export
        verified_only = final_df[final_df["Decision"].isin(["Accept", "Modify"])].copy()
        st.download_button(
            f"⬇️  Download VERIFIED-only ({len(verified_only)} rows)",
            data=to_excel_bytes(verified_only, "IHS_Verified"),
            file_name="IHS_Verified_Output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.markdown("### Preview of verified output")
    st.dataframe(final_df.head(50), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    init_session_state()

    st.title("Consumables Master Data Harmonization  (IHS ↔ NPC ↔ PHC)")
    st.caption("v4 — enhanced fuzzy matching engine + verification workspace")

    with st.expander("ℹ️  What's new in v4?", expanded=False):
        st.markdown("""
**Enhanced fuzzy matching:**
- 🔤 **Medical abbreviation expansion** — `CANN`→cannulated, `DCP`→dynamic compression plate, `TENS`→titanium elastic nail, `KW`→kirschner wire, `LCP`→locking compression plate, +more
- 🎯 **Jaro-Winkler similarity** added to scorer ensemble (catches near-spelling matches)
- 📊 **TF-IDF specificity bonus** — matches sharing rare/informative tokens get +0–10 bonus
- 📏 **Improved suture-size extraction** — now catches bare-digit sizes like `POLYSORB* 0`, `POLYGLACTIN 910 1`

**New: Verification Workspace**
- ✅ Review each match with full context (IHS desc, matched NPC desc, score, notes)
- 📝 Accept / Reject / Modify decisions stored per row
- 🔍 Override NPC codes manually with built-in NPC search
- ⚡ Bulk actions (auto-accept all HIGH confidence, accept/reject filtered subset)
- 📤 Export verified-only output for production use
        """)

    tab_setup, tab_verify, tab_export = st.tabs([
        "1️⃣  Setup & Run", "2️⃣  Verify Matches", "3️⃣  Export Output",
    ])
    with tab_setup:
        render_setup_and_run_tab()
    with tab_verify:
        render_verification_tab()
    with tab_export:
        render_export_tab()


if __name__ == "__main__":
    main()
