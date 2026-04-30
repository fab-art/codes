# Consumables Master Data Harmonization Tool

Streamlit application for harmonizing IHS consumables with NPC and PHC reference datasets.

## Features
- Upload IHS, NPC, and PHC Excel files.
- Dynamic/manual column mapping for schema flexibility.
- Text preprocessing (lowercase, punctuation removal, unit normalization, synonym normalization).
- Aggressive fuzzy matching using RapidFuzz token-set ratio.
- Primary NPC matching, PHC fallback matching, unmatched tagging.
- Confidence scoring, review notes, and downloadable enriched Excel output.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Output columns
The exported `IHS_Enriched_With_NPC.xlsx` includes:
- Original IHS columns
- `NPC_Code`
- `RW_Product_Description`
- `IHBS_Code`
- `Match_Source` (`NPC`, `PHC`, `UNMATCHED`)
- `Match_Score`
- `Confidence_Level`
- `Matched_Candidate_Description`
- `Duplicate_Match_Flag`
- `Review_Notes`
