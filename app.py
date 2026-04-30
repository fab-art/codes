import io
import re
import string
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, process


st.set_page_config(page_title="Consumables Master Data Harmonization", layout="wide")


SYNONYMS = {
    "injector": "syringe",
    "hand gloves": "gloves",
    "iv line": "cannula",
}

CORE_KEYWORDS = {
    "syringe",
    "catheter",
    "screw",
    "gloves",
    "cannula",
    "needle",
    "tube",
    "mask",
    "bandage",
    "gauze",
}


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


UNIT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(mm|ml|cc)\b", flags=re.IGNORECASE)


def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""
    value = str(text).lower().strip()

    for original, canonical in SYNONYMS.items():
        value = value.replace(original, canonical)

    value = value.replace("milliliter", "ml").replace("millilitre", "ml")
    value = value.replace("millimeter", "mm").replace("millimetre", "mm")
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_size_tokens(text: str) -> List[Tuple[float, str]]:
    return [(float(v), u.lower()) for v, u in UNIT_PATTERN.findall(text)]


def core_keyword(text: str) -> Optional[str]:
    tokens = set(text.split())
    for kw in CORE_KEYWORDS:
        if kw in tokens:
            return kw
    return None


def consistency_penalty(source_text: str, candidate_text: str) -> Tuple[float, List[str]]:
    penalty = 0.0
    notes: List[str] = []

    src_sizes = extract_size_tokens(source_text)
    cand_sizes = extract_size_tokens(candidate_text)

    if src_sizes and cand_sizes:
        if src_sizes != cand_sizes:
            penalty += 10
            notes.append("Size mismatch penalty applied")

    src_kw = core_keyword(source_text)
    cand_kw = core_keyword(candidate_text)
    if src_kw and cand_kw and src_kw != cand_kw:
        penalty += 20
        notes.append(f"Keyword mismatch ({src_kw} vs {cand_kw})")

    return penalty, notes


def confidence_level(score: float) -> str:
    if score >= 80:
        return "HIGH"
    if score >= 65:
        return "MEDIUM"
    return "LOW"


def best_match(query: str, reference_norm: List[str], min_score: float) -> Optional[Tuple[int, float]]:
    if not query:
        return None
    result = process.extract(query, reference_norm, scorer=fuzz.token_set_ratio, limit=2)
    if not result:
        return None

    top_text, top_score, top_idx = result[0]
    _ = top_text

    if top_score < min_score:
        return None

    if len(result) > 1 and abs(top_score - result[1][1]) < 2:
        return top_idx, float(top_score)

    return top_idx, float(top_score)


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

    duplicate_flag = False
    review_notes: List[str] = []

    primary = best_match(ihs_desc, npc_norm, min_score=65)
    if primary:
        idx, score = primary
        candidate = npc_norm[idx]
        penalty, notes = consistency_penalty(ihs_desc, candidate)
        final_score = max(0.0, score - penalty)
        review_notes.extend(notes)

        if final_score >= 65:
            return MatchResult(
                npc_code=str(npc_df.iloc[idx][npc_code_col]),
                rw_description=str(npc_df.iloc[idx][npc_desc_col]),
                ihbs_code=None,
                match_source="NPC",
                match_score=round(final_score, 2),
                confidence_level=confidence_level(final_score),
                candidate_desc=str(npc_df.iloc[idx][npc_desc_col]),
                duplicate_flag=duplicate_flag,
                review_notes="; ".join(review_notes),
            )

    secondary = best_match(ihs_desc, phc_norm, min_score=60)
    if secondary:
        idx, score = secondary
        candidate = phc_norm[idx]
        penalty, notes = consistency_penalty(ihs_desc, candidate)
        final_score = max(0.0, score - penalty)
        review_notes.extend(notes)

        if final_score >= 60:
            return MatchResult(
                npc_code=str(phc_df.iloc[idx][phc_code_col]),
                rw_description=str(phc_df.iloc[idx][phc_desc_col]),
                ihbs_code=str(phc_df.iloc[idx][phc_ihbs_col]),
                match_source="PHC",
                match_score=round(final_score, 2),
                confidence_level=confidence_level(final_score),
                candidate_desc=str(phc_df.iloc[idx][phc_desc_col]),
                duplicate_flag=duplicate_flag,
                review_notes="; ".join(review_notes),
            )

    return MatchResult(
        npc_code=None,
        rw_description=None,
        ihbs_code=None,
        match_source="UNMATCHED",
        match_score=0.0,
        confidence_level="LOW",
        candidate_desc=None,
        duplicate_flag=False,
        review_notes="No suitable match found",
    )


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
        mapping[map_key] = st.selectbox(f"{label}: {human_name}", cols, key=f"{label}-{map_key}")
    return mapping


def main() -> None:
    st.title("Consumables Master Data Harmonization Tool (IHS ↔ NPC ↔ PHC)")

    c1, c2, c3 = st.columns(3)
    with c1:
        ihs_file = st.file_uploader("Upload IHS File", type=["xlsx", "xls"], key="ihs")
    with c2:
        npc_file = st.file_uploader("Upload NPC File", type=["xlsx", "xls"], key="npc")
    with c3:
        phc_file = st.file_uploader("Upload PHC File", type=["xlsx", "xls"], key="phc")

    if not (ihs_file and npc_file and phc_file):
        st.info("Upload all three Excel files to continue.")
        return

    ihs_df = pd.read_excel(ihs_file)
    npc_df = pd.read_excel(npc_file)
    phc_df = pd.read_excel(phc_file)

    st.markdown("### Dataset Preview")
    tab1, tab2, tab3 = st.tabs(["IHS", "NPC", "PHC"])
    with tab1:
        st.dataframe(ihs_df.head(10), use_container_width=True)
    with tab2:
        st.dataframe(npc_df.head(10), use_container_width=True)
    with tab3:
        st.dataframe(phc_df.head(10), use_container_width=True)

    st.markdown("### Column Mapping")

    ihs_map = mapping_ui(ihs_df, "IHS", {"Product Name / Description": "desc"})
    npc_map = mapping_ui(
        npc_df,
        "NPC",
        {
            "NPC Code": "npc_code",
            "RW Product Description": "rw_desc",
        },
    )
    phc_map = mapping_ui(
        phc_df,
        "PHC",
        {
            "Product Description": "desc",
            "NPC Code": "npc_code",
            "IHBS Code": "ihbs_code",
        },
    )

    if st.button("Run Matching", type="primary"):
        all_maps = [ihs_map["desc"], npc_map["npc_code"], npc_map["rw_desc"], phc_map["desc"], phc_map["npc_code"], phc_map["ihbs_code"]]
        if any(not m for m in all_maps):
            st.error("Please map all required columns before running matching.")
            return

        progress = st.progress(0)

        ihs_work = ihs_df.copy()
        npc_work = npc_df.copy()
        phc_work = phc_df.copy()

        ihs_work["_norm"] = ihs_work[ihs_map["desc"]].apply(normalize_text)
        npc_work["_norm"] = npc_work[npc_map["rw_desc"]].apply(normalize_text)
        phc_work["_norm"] = phc_work[phc_map["desc"]].apply(normalize_text)

        results = []
        total = len(ihs_work)
        for i, row in ihs_work.iterrows():
            match = match_ihs_row(
                ihs_desc=row["_norm"],
                npc_df=npc_work,
                phc_df=phc_work,
                npc_norm_col="_norm",
                phc_norm_col="_norm",
                npc_code_col=npc_map["npc_code"],
                npc_desc_col=npc_map["rw_desc"],
                phc_code_col=phc_map["npc_code"],
                phc_desc_col=phc_map["desc"],
                phc_ihbs_col=phc_map["ihbs_code"],
            )
            results.append(match)
            progress.progress(min((i + 1) / max(total, 1), 1.0))

        result_df = ihs_work.drop(columns=["_norm"]).copy()
        result_df["NPC_Code"] = [r.npc_code for r in results]
        result_df["RW_Product_Description"] = [r.rw_description for r in results]
        result_df["IHBS_Code"] = [r.ihbs_code for r in results]
        result_df["Match_Source"] = [r.match_source for r in results]
        result_df["Match_Score"] = [r.match_score for r in results]
        result_df["Confidence_Level"] = [r.confidence_level for r in results]
        result_df["Matched_Candidate_Description"] = [r.candidate_desc for r in results]
        result_df["Duplicate_Match_Flag"] = [r.duplicate_flag for r in results]
        result_df["Review_Notes"] = [r.review_notes for r in results]

        match_rate = (result_df["Match_Source"] != "UNMATCHED").mean() * 100 if len(result_df) else 0
        st.success(f"Matching complete. Combined match rate: {match_rate:.2f}%")

        st.markdown("### Enriched Output Preview")
        st.dataframe(result_df.head(50), use_container_width=True)

        st.download_button(
            "Download IHS_Enriched_With_NPC.xlsx",
            data=to_excel_bytes(result_df),
            file_name="IHS_Enriched_With_NPC.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
