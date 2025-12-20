"""Central configuration for RAG1-Mini pipeline.

Contains:
- Project paths for all pipeline stages (extraction through embedding)
- Text cleaning patterns (line removal, inline removal, substitutions)
- NLP settings (spaCy model, sentence filtering)
- Chunking parameters (token limits, overlap)
- Embedding settings (API configuration via .env)
- Weaviate vector database settings
"""
import re
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
import os

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Processing pipeline directories
DIR_RAW_EXTRACT = DATA_DIR / "processed" / "01_raw_extraction"
DIR_MANUAL_REVIEW = DATA_DIR / "processed" / "02_manual_review"
DIR_DEBUG_CLEAN = DATA_DIR / "processed" / "03_markdown_cleaning"
DIR_NLP_CHUNKS = DATA_DIR / "processed" / "04_nlp_chunks"
DIR_FINAL_CHUNKS = DATA_DIR / "processed" / "05_final_chunks"
DIR_EMBEDDINGS = DATA_DIR / "processed" / "06_embeddings"

# Logging
DIR_CLEANING_LOGS = DATA_DIR / "logs"
DIR_CLEANING_LOGS.mkdir(parents=True, exist_ok=True)
CLEANING_LOG_FILE = DIR_CLEANING_LOGS / "cleaning_report.log"


# ============================================================================
# CLEANING PATTERNS
# ============================================================================

# Pattern format: (regex_pattern, pattern_name)
# All patterns use explicit case matching in regex (no re.IGNORECASE needed)

LINE_REMOVAL_PATTERNS: List[Tuple[str, str]] = [
    # Figure/Table captions: "Figure 2. Model diagram", "Table 1-3: Flow Chart"
    # Matches: Figure/Fig/Table/Tab + number/identifier + UPPERCASE start
    # Preserves: "Figure 2 shows..." (lowercase after number)
    (
        r'^\s*(#+\s*)?([Ff][Ii][Gg]([Uu][Rr][Ee])?|[Tt][Aa][Bb]([Ll][Ee])?)\.?\s+[\w\.\-]+\s+[A-Z]',
        'FIGURE_TABLE_CAPTION'
    ),
    
    # Learning objectives: "LO 1.2", "LO 5"
    (
        r'^\s*(##\s*)?LO\s+\d',
        'LEARNING_OBJECTIVE'
    ),
    
    # Single character lines: isolated letters, numbers, symbols
    (
        r'^\s*[a-zA-Z0-9\.\|\-]\s*$',
        'SINGLE_CHAR'
    ),
    
    # Heading with only a number: "## 5"
    (
        r'^\s*##\s+\d+\s*$',
        'HEADING_SINGLE_NUMBER'
    ),
]


INLINE_REMOVAL_PATTERNS: List[Tuple[str, str]] = [
    # Figure/table references in parentheses: "(Figure 2)", "(TABLE 1-3)"
    (
        r'\(\s*([Ff][Ii][Gg]([Uu][Rr][Ee])?|[Tt][Aa][Bb]([Ll][Ee])?)\.?\s*[\d\.\-]+[a-zA-Z]?\s*\)',
        'FIG_TABLE_REF'
    ),
    
    # Footnote markers: "fn3", "fn12" (typically appear mid-sentence)
    (
        r'\bfn\d+\b\s*',
        'FOOTNOTE_MARKER'
    ),
    
    # Standalone numbers after punctuation: ". 81 We" -> ". We"
    # Removes page numbers and footnote references
    (
        r'(?<=[.!?\"\'])\s+\d+\s+(?=[A-Z])',
        'TRAILING_NUMBER'
    ),
]


CHARACTER_SUBSTITUTIONS: List[Tuple[str, str, str]] = [
    # Format: (old_string, new_string, substitution_name)
    ('/u2014.d', '--', 'EM_DASH_WITH_SUFFIX'),
    ('/u2014', '--', 'EM_DASH'),
    ('&', '&', 'HTML_AMPERSAND'),
]


# List marker pattern: used in special processing function
LIST_MARKER_PATTERN = r'^\s*\([a-z]\)\s+'

# Punctuation for paragraph merging decisions
TERMINAL_PUNCTUATION = ('.', '!', '?', ':', ';', '"', '"')
SENTENCE_ENDING_PUNCTUATION = ('.', '!', '?', '"', '"')

# Report formatting
REPORT_WIDTH = 100


# ============================================================================
# NLP SETTINGS
# ============================================================================

SPACY_MODEL = "en_core_sci_sm"

# Valid sentence endings for filtering
VALID_ENDINGS = ('.', '?', '!', '"', '"', ')', ']')

# Sentence filtering
MIN_SENTENCE_WORDS = 2

# Markdown header detection
HEADER_CHAPTER = '# '
HEADER_SECTION = '##'

# Context string formatting
CONTEXT_SEPARATOR = ' > '


# ============================================================================
# CHUNKING SETTINGS
# ============================================================================


# Chunking parameters (tuned for text-embedding-3-large)
MAX_CHUNK_TOKENS = 800
MAX_SENTENCE_TOKENS = 800

# Tokenizer model name (OpenAI compatible)
TOKENIZER_MODEL = "text-embedding-3-large"

# Configurable overlap: number of sentences to carry from previous chunk
OVERLAP_SENTENCES = 2  # Adjust this value (0 = no overlap, 2-3 recommended)

# Chunk ID formatting
CHUNK_ID_PREFIX = 'chunk'
CHUNK_ID_SEPARATOR = '::'

# Safety limits for chunking loops
MAX_LOOP_ITERATIONS = 1000


# ============================================================================
# EMBEDDING SETTINGS
# ============================================================================


# Embedding model (OpenAI / OpenRouter compatible)
EMBEDDING_MODEL = TOKENIZER_MODEL  # "text-embedding-3-large"

# Safety limits
MAX_BATCH_TOKENS = 12_000   # conservative batch size
MAX_RETRIES = 3

# Load environment variables from the .env file (in src/ directory)
load_dotenv(Path(__file__).parent / ".env")

# Now you can access the variables
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL')
EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID')


# ============================================================================
# WEAVIATE SETTINGS
# ============================================================================

# Connection settings
WEAVIATE_HOST = os.getenv('WEAVIATE_HOST', 'localhost')
WEAVIATE_HTTP_PORT = int(os.getenv('WEAVIATE_HTTP_PORT', '8080'))
WEAVIATE_GRPC_PORT = int(os.getenv('WEAVIATE_GRPC_PORT', '50051'))

# Batch upload settings
WEAVIATE_BATCH_SIZE = 100  # Objects per batch (Weaviate recommends 100-1000)

# Collection naming components (auto-generated)
# Format: RAG_{chunking_strategy}_{embedding_model_short}_{version}
CHUNKING_STRATEGY_NAME = "section800"  # Describes current chunking approach
EMBEDDING_MODEL_SHORT = "embed3large"  # Short name for embedding model
COLLECTION_VERSION = "v1"  # Increment when re-running with same strategy


def get_collection_name() -> str:
    """
    Generate collection name from current pipeline configuration.

    Returns:
        Collection name in format: RAG_{strategy}_{model}_{version}

    Example:
        "RAG_section800_embed3large_v1"
    """
    return f"RAG_{CHUNKING_STRATEGY_NAME}_{EMBEDDING_MODEL_SHORT}_{COLLECTION_VERSION}"


# ============================================================================
# UI SETTINGS
# ============================================================================

# Book categories for UI display (organized by domain)
BOOK_CATEGORIES = {
    "neuroscience": [
        "Behave, The_Biology of Humans at Our Best Worst (Robert M. Sapolsky)",
        "Biopsychology (John Pinel, Steven Barnes)",
        "Brain and behavior, a cognitive neuroscience perspective (David Eagleman, Jonathan Downar)",
        "Cognitive Biology , Evolutionary and Developmental Perspectives on Mind Brain and Behavior (Luca Tommasi, Mary A.Peterson, Lynn Nadel)",
        "Cognitive Neuroscience, The Biology of the Mind (Michael Gazzaniga)",
        "Determined, a science of life without free will (Robert M. Sapolsky)",
        "Fundamentals of Cognitive Neuroscience,  A_Beginners Guide(Nicole M. Gage Bernard)",
        "Psychobiology of Behaviour (Konstanthos N,Fountoulakis, Loannis Nimatoudis)",
    ],
    "philosophy": [
        "Essays and Aphorisms (Arthur Schopenhauer)",
        "Letters from a Stoic (Seneca)",
        "Tao te ching Lao_tzu (Lao Tzu)",
        "The Analects Conclusions and Conversations (Confucius)",
        "The Art of Living ,The Classical Manual on Virtue Happiness and Effectiveness (Epictetus)",
        "The Enchiridion (Epictetus)",
        "The Meditations (Marcus Aurelius)",
        "The Pocket Oracle and Art of Prudence (Baltasar Gracian)",
        "The essays, counsels and maxims (Arthur Schopenhauer)",
        "Thinking Fast and Slow (Daniel Kahneman)",
        "Wisdom of Life (Schopenhauer)",
    ],
}

# Search UI defaults
DEFAULT_TOP_K = 10
MAX_TOP_K = 20


# ============================================================================
# EVALUATION SETTINGS (RAGAS)
# ============================================================================

# Model selection based on research (see memory-bank/model-selection.md)
# Generation model: Fast, cost-effective for answer generation
EVAL_GENERATION_MODEL = "openai/gpt-5-mini"

# Evaluation model: High quality for LLM-as-judge (RAGAS metrics)
EVAL_EVALUATION_MODEL = "anthropic/claude-haiku-4.5"

# Test questions file location
EVAL_TEST_QUESTIONS_FILE = PROJECT_ROOT / "src" / "evaluation" / "test_questions.json"

# Results output directory
EVAL_RESULTS_DIR = DATA_DIR / "evaluation" / "results"


# ============================================================================
# QUERY PREPROCESSING SETTINGS
# ============================================================================

# Model for query classification and step-back prompting
# Using fast, cheap model since these are simple classification tasks
PREPROCESSING_MODEL = "openai/gpt-5-nano"


# ============================================================================
# ANSWER GENERATION SETTINGS
# ============================================================================

# Default model for answer generation (balanced quality/cost)
GENERATION_MODEL = "openai/gpt-5-mini"

# Available models for UI selection (ordered by cost)
# See memory-bank/model-selection.md for pricing details
AVAILABLE_GENERATION_MODELS = [
    ("openai/gpt-5-nano", "GPT-5 Nano ($0.05/$0.40 per 1M) - Budget"),
    ("deepseek/deepseek-chat", "DeepSeek V3.2 ($0.28/$0.42 per 1M) - Value"),
    ("openai/gpt-5-mini", "GPT-5 Mini ($0.25/$2.00 per 1M) - Balanced"),
    ("google/gemini-3-flash", "Gemini 3 Flash ($0.50/$3.00 per 1M) - Quality"),
    ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5 ($1.00/$5.00 per 1M) - Premium"),
]

# Enable/disable answer generation globally (can be overridden in UI)
ENABLE_ANSWER_GENERATION = True

# Enable/disable query preprocessing (classification + step-back)
ENABLE_QUERY_PREPROCESSING = True