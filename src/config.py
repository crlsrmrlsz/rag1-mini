import re
from pathlib import Path

# --- PROJECT PATHS ---
# Assumes this file is in src/config.py, so project root is one level up
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Phase 1 Output / Phase 2 Input
DIR_RAW_EXTRACT = DATA_DIR / "processed" / "01_raw_extraction"
# Phase 2 Output / Phase 3 Input (User places reviewed files here)
DIR_MANUAL_REVIEW = DATA_DIR / "processed" / "02_manual_review"
# Phase 3 Debug Output
DIR_DEBUG_CLEAN = DATA_DIR / "processed" / "03_structural_debug"
# Phase 5 Final Output
DIR_FINAL_CHUNKS = DATA_DIR / "processed" / "04_final_chunks"

# --- REGEX PATTERNS (Phase 3) ---
#
LINE_ARTIFACT_PATTERNS = [
    (r'^\s*(FIGURE|FIG|Fig|TABLE|TAB|Tab)(\.?)\s*[\d\.\-]+\s*.*$', ''),
    (r'^\s*(Source|Credit|Data from):.*$', ''),
]

INLINE_REMOVAL_PATTERNS = [
    r'\(\s*(FIGURE|FIG|Fig|TABLE|TAB|Tab)\.?\s*[\d\.\-]+\s*\)',
]

# --- NLP SETTINGS (Phase 4 & 5) ---
SPACY_MODEL = "en_core_sci_sm"

# Valid terminal punctuation for sentence filtering
VALID_ENDINGS = ('.', '?', '!', '"', '”', ')', ']')