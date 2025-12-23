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


def get_collection_name(chunking_strategy: str = None) -> str:
    """
    Generate collection name from pipeline configuration.

    Args:
        chunking_strategy: Chunking strategy name (e.g., "section", "semantic").
            If None, uses CHUNKING_STRATEGY_NAME from config.

    Returns:
        Collection name in format: RAG_{strategy}_{model}_{version}

    Example:
        >>> get_collection_name()  # Uses config default
        "RAG_section800_embed3large_v1"
        >>> get_collection_name("semantic")
        "RAG_semantic_embed3large_v1"
    """
    strategy = chunking_strategy if chunking_strategy else CHUNKING_STRATEGY_NAME
    strategy_safe = strategy.replace(".", "_")
    return f"RAG_{strategy_safe}_{EMBEDDING_MODEL_SHORT}_{COLLECTION_VERSION}"


# ============================================================================
# UI SETTINGS
# ============================================================================

# Search UI defaults
DEFAULT_TOP_K = 10
MAX_TOP_K = 20


# ============================================================================
# EVALUATION SETTINGS (RAGAS)
# ============================================================================

# Model selection based on research (see memory-bank/model-selection.md)
# Generation model: Fast, cost-effective for answer generation
EVAL_GENERATION_MODEL = "google/gemini-3-flash-preview"

# Evaluation model: Cost-effective for LLM-as-judge (RAGAS metrics)
# Claude 3 Haiku is 75% cheaper than Haiku 4.5 ($0.25/$1.25 vs $1/$5 per 1M tokens)
# Sufficient for binary evaluation judgments (claim verification, relevance checks)
EVAL_EVALUATION_MODEL = "anthropic/claude-3-haiku"

# Test questions file location
EVAL_TEST_QUESTIONS_FILE = PROJECT_ROOT / "src" / "evaluation" / "test_questions.json"

# Results output directory
EVAL_RESULTS_DIR = DATA_DIR / "evaluation" / "ragas_results"


# ============================================================================
# QUERY PREPROCESSING SETTINGS
# ============================================================================

# Corpus topics for query grounding (helps LLM generate vocabulary in the corpus)
# This lightweight list (~50 tokens) guides preprocessing without prompt bloat
# Covers: neuroscience (Sapolsky, Kahneman), Stoicism, Taoism, Confucianism,
# German pessimism (Schopenhauer), Baroque prudence (Gracián)
CORPUS_TOPICS = (
    "neuroscience, cognitive psychology, behavioral biology, brain mechanisms, "
    "decision-making, emotions, stress, memory, aggression, altruism, free will, "
    "Stoic philosophy, Taoism, Confucianism, German pessimism, Baroque prudence, "
    "virtue ethics, practical wisdom, cognitive biases, the will, suffering"
)

# Model for query preprocessing (step-back, multi-query, decomposition)
# Using fast, cheap model since these are simple transformation tasks
PREPROCESSING_MODEL = "deepseek/deepseek-v3.2"

# Fallback models for preprocessing (used if dynamic fetch fails)
# These are updated manually when OpenRouter availability changes
AVAILABLE_PREPROCESSING_MODELS = [
    ("deepseek/deepseek-v3.2", "Budget: DeepSeek V3.2"),
    ("google/gemini-3-flash-preview", "Value: Gemini 3 Flash"),
    ("anthropic/claude-haiku-4.5", "Quality: Claude Haiku 4.5"),
    ("anthropic/claude-opus-4.5", "Premium: Claude Opus 4.5"),
]


# ============================================================================
# ANSWER GENERATION SETTINGS
# ============================================================================

# Default model for answer generation (balanced quality/cost)
GENERATION_MODEL = "google/gemini-3-flash-preview"

# Fallback models for generation (used if dynamic fetch fails)
AVAILABLE_GENERATION_MODELS = [
    ("deepseek/deepseek-v3.2", "Budget: DeepSeek V3.2"),
    ("google/gemini-3-flash-preview", "Value: Gemini 3 Flash"),
    ("anthropic/claude-haiku-4.5", "Quality: Claude Haiku 4.5"),
    ("anthropic/claude-opus-4.5", "Premium: Claude Opus 4.5"),
]

# Enable/disable answer generation globally (can be overridden in UI)
ENABLE_ANSWER_GENERATION = True

# Enable/disable query preprocessing (strategy-based transformation)
ENABLE_QUERY_PREPROCESSING = True


# ============================================================================
# PREPROCESSING STRATEGY SETTINGS
# ============================================================================

# Available preprocessing strategies
# Format: (strategy_id, display_label, description)
AVAILABLE_PREPROCESSING_STRATEGIES = [
    ("none", "None", "No preprocessing, use original query"),
    ("step_back", "Step-Back", "Transform to broader concepts for better retrieval"),
    ("multi_query", "Multi-Query", "Generate 4 targeted queries + RRF merge"),
    ("decomposition", "Decomposition", "Break into sub-questions + RRF merge"),
]

# Default strategy for UI and preprocess_query() when not specified
DEFAULT_PREPROCESSING_STRATEGY = "step_back"


# ============================================================================
# CHUNKING STRATEGY SETTINGS
# ============================================================================

# Available chunking strategies
# Format: (strategy_id, display_label, description)
AVAILABLE_CHUNKING_STRATEGIES = [
    ("section", "Section (Baseline)", "Sequential with sentence overlap, respects markdown sections"),
    ("semantic", "Semantic", "Embedding similarity-based boundaries for topic coherence"),
    ("contextual", "Contextual", "LLM-generated chunk context (Anthropic-style, +35% improvement)"),
    # Future strategies (uncomment when implemented):
    # ("raptor", "RAPTOR", "Hierarchical summarization tree"),
]

# Default strategy for CLI when not specified
DEFAULT_CHUNKING_STRATEGY = "section"

# Semantic chunking parameters
# Threshold for cosine similarity between adjacent sentences
# Lower = fewer splits (larger chunks), Higher = more splits (smaller chunks)
#
# References:
# - arXiv:2410.13070 (Oct 2024): Tested absolute thresholds [0.1-0.5], found
#   absolute thresholds more consistent than percentile-based across corpus sizes
# - LlamaIndex/LangChain: Use 95th percentile of cosine distances (Kamradt method)
# - Chroma Research: Excerpt relevance filtering at 0.40-0.43 cosine similarity
#
# Tuning notes:
# - 0.75: Too aggressive (small chunks, fragmented topics)
# - 0.5: Conservative, major topic shifts only
# - 0.4: Recommended default (aligns with Chroma 0.40-0.43 range)
# - 0.3: Test value for maximum context grouping
# Note: MAX_CHUNK_TOKENS (800) limits chunk size regardless of threshold
SEMANTIC_SIMILARITY_THRESHOLD = 0.4  # Default; test 0.3 for larger groups

# Contextual chunking parameters (Anthropic-style)
# Model for generating contextual snippets (fast, cheap model recommended)
CONTEXTUAL_MODEL = "anthropic/claude-3-haiku"

# Number of neighboring chunks to include as context for LLM
CONTEXTUAL_NEIGHBOR_CHUNKS = 2  # chunks before + after current chunk

# Maximum tokens for the contextual snippet (output limit)
CONTEXTUAL_MAX_SNIPPET_TOKENS = 100

# Prompt template for generating contextual snippets
# Placeholders: {document_context}, {chunk_text}, {book_name}, {context_path}
CONTEXTUAL_PROMPT = """<document>
{document_context}
</document>

Here is the chunk we want to situate within the document:
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context (2-3 sentences) to situate this chunk within the overall document.
The context should help a reader understand what broader topic or argument this relates to.
Include key terms or entities that provide disambiguation.

Book: "{book_name}"
Section: "{context_path}"

Answer only with the contextual description, nothing else."""


def get_semantic_folder_name(threshold: float = SEMANTIC_SIMILARITY_THRESHOLD) -> str:
    """Generate semantic chunking folder name with threshold.

    Creates folder names like 'semantic_0.5' or 'semantic_0.75' to distinguish
    outputs from different threshold configurations.

    Args:
        threshold: Similarity threshold (0.0-1.0).

    Returns:
        Folder name in format 'semantic_{threshold}'.

    Example:
        >>> get_semantic_folder_name(0.5)
        'semantic_0.5'
        >>> get_semantic_folder_name(0.75)
        'semantic_0.75'
    """
    # Format threshold: remove trailing zeros (0.50 -> 0.5, 0.75 -> 0.75)
    threshold_str = f"{threshold:.2f}".rstrip("0").rstrip(".")
    return f"semantic_{threshold_str}"


# ============================================================================
# STRATEGY METADATA REGISTRY
# ============================================================================
# Central registry for chunking strategy metadata, used for:
# - UI display (descriptions, labels)
# - Strategy-scoped embedding paths
# - Collection discovery and enrichment

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class StrategyMetadata:
    """Metadata for a chunking strategy.

    Attributes:
        key: Strategy identifier (e.g., "section", "semantic_0.5").
        display_name: Human-readable name for UI display.
        description: Short description of the strategy's approach.
    """
    key: str
    display_name: str
    description: str


# Registry of known chunking strategies with their metadata
STRATEGY_REGISTRY: Dict[str, StrategyMetadata] = {
    "section": StrategyMetadata(
        key="section",
        display_name="Section-Based Chunking",
        description="Preserves document structure with sentence overlap",
    ),
    "contextual": StrategyMetadata(
        key="contextual",
        display_name="Contextual Chunking",
        description="LLM-generated context prepended (+35% improvement)",
    ),
    # Semantic strategies are generated dynamically based on threshold
}


def get_strategy_metadata(strategy: str) -> StrategyMetadata:
    """Get metadata for a chunking strategy.

    Args:
        strategy: Strategy key (e.g., "section", "semantic_0.5", "contextual").

    Returns:
        StrategyMetadata with display name and description.
        For unknown strategies, generates a generic fallback.

    Example:
        >>> get_strategy_metadata("section")
        StrategyMetadata(key='section', display_name='Section-Based Chunking', ...)
        >>> get_strategy_metadata("semantic_0.5")
        StrategyMetadata(key='semantic_0.5', display_name='Semantic Chunking (0.5)', ...)
    """
    # Check registry first
    if strategy in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY[strategy]

    # Handle semantic_X.X variants dynamically
    if strategy.startswith("semantic_"):
        threshold = strategy.split("_", 1)[1]
        return StrategyMetadata(
            key=strategy,
            display_name=f"Semantic Chunking ({threshold})",
            description=f"Embedding similarity boundaries (threshold: {threshold})",
        )

    # Fallback for unknown strategies
    return StrategyMetadata(
        key=strategy,
        display_name=strategy.replace("_", " ").title(),
        description="Custom chunking strategy",
    )


def get_embedding_folder_path(strategy: str) -> Path:
    """Get strategy-scoped embedding folder path.

    Creates isolated embedding storage per strategy, enabling A/B testing
    of different chunking approaches without data overwrites.

    Args:
        strategy: Strategy key (e.g., "section", "semantic_0.5", "contextual").

    Returns:
        Path to embedding folder: data/processed/06_embeddings/{strategy}/

    Example:
        >>> get_embedding_folder_path("section")
        PosixPath('.../data/processed/06_embeddings/section')
        >>> get_embedding_folder_path("semantic_0.5")
        PosixPath('.../data/processed/06_embeddings/semantic_0.5')
    """
    # Sanitize strategy to prevent path traversal
    # Replace all path separators and multiple dots with underscores
    import re
    safe_strategy = re.sub(r'[/\\]+', '_', strategy)  # Replace path separators
    safe_strategy = re.sub(r'\.{2,}', '_', safe_strategy)  # Replace multiple dots
    return DIR_EMBEDDINGS / safe_strategy