# Content Preparation

> **Stages:** 1-2 of 8 | Converts PDFs to clean Markdown ready for chunking

This document covers the complete workflow for preparing text content from PDF books: pre-cleaning, PDF-to-markdown conversion, and automated text cleaning.

## Overview

Content preparation follows three phases:

```
Phase 1: PDF Pre-Cleaning (manual)
    ↓
Phase 2: PDF to Markdown (Docling)
    ↓
Phase 3: Markdown Cleaning (automated + manual review)
```

Each phase addresses specific challenges encountered with complex academic texts, particularly neuroscience books with dense layouts.

---

## Phase 1: PDF Pre-Cleaning

Before any automated extraction, PDFs are manually cleaned using PDF editing tools (e.g., Adobe Acrobat, PDF-XChange Editor).

### Elements Removed

| Element | Why Remove |
|---------|------------|
| References/Bibliography | Dense citation blocks confuse layout detection |
| Glossary | Alphabetical lists extract poorly |
| Index | Page number lists are noise |
| Acknowledgments | Not core content |
| Notes sections | Often formatted as footnotes |
| Appendices | Supplementary material, separate handling needed |

### Why This Matters

Neuroscience textbooks present unique extraction challenges:
- **Multi-column layouts** with figures spanning columns
- **Case study boxes** highlighted in different colors
- **Inline equations** mixed with body text
- **Dense figure placement** interrupting paragraphs

Pre-cleaning reduces the complexity that automated tools must handle, significantly improving downstream extraction quality.

---

## Phase 2: PDF to Markdown Conversion

### Initial Approach: PyMuPDF4LLM

First attempts used [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/), a lightweight PDF-to-markdown converter.

```python
import pymupdf4llm

# Basic conversion
md_text = pymupdf4llm.to_markdown(pdf_file)

# With header/footer removal
md_text = pymupdf4llm.to_markdown(pdf_file, header=False, footer=False)
```

**How PyMuPDF4LLM detects layout:**
- Extracts text blocks with coordinates from PDF structure
- Uses heuristics based on font size, position, and spacing
- Optional [PyMuPDF-Layout](https://pymupdf.readthedocs.io/en/latest/pymupdf-layout/index.html) extension adds AI-based detection (1.3-1.8M parameters)
- K-means clustering can detect column structures

**Challenges with neuroscience books:**

| Issue | Description |
|-------|-------------|
| Multi-column merging | Text blocks from different columns merged incorrectly |
| Figure interruption | Reading order jumbled when figures appear mid-paragraph |
| Case study boxes | Highlighted sections extracted out of order or merged with surrounding text |
| Tables | Extracted as garbled text, losing all structure |
| Mathematical notation | Unicode escape sequences, garbled symbols |
| Footnotes | Mixed into body text, breaking sentence flow |

These issues required extensive manual cleanup, making PyMuPDF4LLM impractical for our corpus.

### Solution: Docling

[Docling](https://github.com/docling-project/docling) (IBM Research) uses AI vision models for layout understanding, solving the problems PyMuPDF4LLM couldn't handle.

**How Docling differs:**

| Aspect | PyMuPDF4LLM | Docling |
|--------|-------------|---------|
| Layout method | PDF coordinates + heuristics | Vision AI model (DocLayNet) |
| Parameters | 1.3-1.8M | 20M+ |
| Speed | Fast (~1 sec/page) | Slower (~4 sec/page) |
| Tables | Poor extraction | Excellent structure preservation |
| Complex layouts | Struggles with multi-column, figures | Handles well |
| Element classification | Manual regex | Native labels (CAPTION, FOOTNOTE, etc.) |

**Key Docling advantages:**
- **DocLayNet**: Vision model trained on diverse document layouts
- **TableFormer**: Dedicated model for table structure recognition
- **Native element labels**: Automatically classifies CAPTION, FOOTNOTE, TABLE, PICTURE, etc.
- **Reading order**: Correctly sequences text across complex layouts

### Implementation

```python
# src/content_preparation/extraction/docling_parser.py

from docling.datamodel.document import InputFormat, DocItemLabel
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

def extract_pdf(pdf_path: Path) -> str:
    """Extract text from PDF to markdown."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False           # Skip OCR (text PDFs only)
    pipeline_options.do_table_structure = False  # Skip table parsing

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(pdf_path)
    doc = result.document

    # Remove artifacts using native labels
    labels_to_remove = {
        DocItemLabel.CAPTION,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.TABLE
    }

    items_to_remove = [
        item for item, level in doc.iterate_items()
        if hasattr(item, "label") and item.label in labels_to_remove
    ]
    doc.delete_items(node_items=items_to_remove)

    # Also remove pictures and their children (diagram labels, etc.)
    # ... (see full implementation in docling_parser.py)

    return doc.export_to_markdown()
```

### Removed Elements

| Element | Why Removed |
|---------|-------------|
| `CAPTION` | Figure captions without figures are noise |
| `FOOTNOTE` | Disrupts text flow, often references |
| `PAGE_HEADER` | Running headers with chapter titles |
| `PAGE_FOOTER` | Page numbers, publisher info |
| `TABLE` | Complex structure requires separate handling |
| `PICTURE` | Images don't translate to text; children (labels like "a", "b") removed too |

---

## Phase 3: Markdown Cleaning

After Docling extraction, two cleaning steps refine the output.

### Manual Review (Optional)

Location: `data/processed/02_manual_review/`

Purpose: Catch extraction errors before automated cleaning:
- Verify heading hierarchy
- Fix obvious extraction failures
- Remove any remaining artifacts

### Automated Cleaning

The automated cleaner applies regex patterns to remove common artifacts.

#### Line Removal Patterns

Full lines matching these patterns are deleted:

| Pattern | Example Match | Purpose |
|---------|---------------|---------|
| `FIGURE_TABLE_CAPTION` | "Figure 2. Model diagram" | Remove orphaned captions |
| `LEARNING_OBJECTIVE` | "LO 1.2" | Remove textbook learning objectives |
| `SINGLE_CHAR` | "a" (isolated line) | Remove OCR noise, diagram labels |
| `HEADING_SINGLE_NUMBER` | "## 5" | Remove meaningless number-only headings |

```python
# From src/config.py
LINE_REMOVAL_PATTERNS = [
    # Figure/Table captions starting with uppercase after number
    (r'^\s*(#+\s*)?([Ff][Ii][Gg]([Uu][Rr][Ee])?|[Tt][Aa][Bb]([Ll][Ee])?)\.?\s+[\w\.\-]+\s+[A-Z]',
     'FIGURE_TABLE_CAPTION'),

    # Learning objectives
    (r'^\s*(##\s*)?LO\s+\d', 'LEARNING_OBJECTIVE'),

    # Single character lines
    (r'^\s*[a-zA-Z0-9\.\|\-]\s*$', 'SINGLE_CHAR'),

    # Headings with only numbers
    (r'^\s*##\s+\d+\s*$', 'HEADING_SINGLE_NUMBER'),
]
```

#### Inline Removal Patterns

Text within lines matching these patterns is removed:

| Pattern | Example | Purpose |
|---------|---------|---------|
| `FIG_TABLE_REF` | "(Figure 2)" | Remove parenthetical figure references |
| `FOOTNOTE_MARKER` | "fn3" | Remove footnote markers mid-sentence |
| `TRAILING_NUMBER` | ". 81 We" → ". We" | Remove page numbers between sentences |

```python
INLINE_REMOVAL_PATTERNS = [
    # Figure/table references in parentheses
    (r'\(\s*([Ff][Ii][Gg]([Uu][Rr][Ee])?|[Tt][Aa][Bb]([Ll][Ee])?)\.?\s*[\d\.\-]+[a-zA-Z]?\s*\)',
     'FIG_TABLE_REF'),

    # Footnote markers
    (r'\bfn\d+\b\s*', 'FOOTNOTE_MARKER'),

    # Page numbers after sentence punctuation
    (r'(?<=[.!?\"\'])\s+\d+\s+(?=[A-Z])', 'TRAILING_NUMBER'),
]
```

#### Character Substitutions

| Original | Replacement | Purpose |
|----------|-------------|---------|
| `/u2014` | `--` | Unicode em-dash escape sequence |
| `&amp;` | `&` | HTML entity |

#### Advanced Cleaning Features

**Paragraph Consolidation**: Merges incorrectly split paragraphs based on punctuation:
```
Input:  "The brain controls\n\nbehavior through"
Output: "The brain controls behavior through"
```

**List Marker Removal**: Removes "(a)", "(b)" markers that follow terminal punctuation.

**Hyphenated Word Recovery**: Fixes line-break hyphenation:
```
Input:  "mu- opioid receptor"
Output: "mu-opioid receptor"
```

---

## Data Flow

```
data/raw/{corpus}/*.pdf (pre-cleaned manually)
    │
    ▼ Stage 1: extract_pdf()
data/processed/01_raw_extraction/{corpus}/*.md
    │
    ▼ Manual review (optional)
data/processed/02_manual_review/{corpus}/*.md
    │
    ▼ Stage 2: run_structural_cleaning()
data/processed/03_markdown_cleaning/{corpus}/*.md
    │
    ▼ Stage 3: NLP segmentation
data/processed/04_nlp_chunks/{corpus}/*.json
```

### Running the Stages

```bash
# Stage 1: PDF to Markdown
python -m src.stages.run_stage_1_extraction

# Stage 2: Automated cleaning (reads from 02_manual_review/)
python -m src.stages.run_stage_2_processing
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/content_preparation/extraction/docling_parser.py` | Docling PDF extraction |
| `src/content_preparation/cleaning/text_cleaner.py` | Regex cleaning orchestration |
| `src/config.py` (lines 45-109) | All regex patterns and thresholds |
| `src/stages/run_stage_1_extraction.py` | Stage 1 CLI runner |
| `src/stages/run_stage_2_processing.py` | Stage 2 CLI runner |

---

## Lessons Learned

1. **Pre-cleaning is essential**: Removing references, glossaries, and appendices before extraction prevents layout detection failures.

2. **Layout detection matters**: Coordinate-based methods (PyMuPDF4LLM) fail on complex academic layouts. Vision-based models (Docling) handle multi-column, figures, and boxes correctly.

3. **Speed vs. accuracy tradeoff**: Docling is ~4x slower than PyMuPDF4LLM but produces dramatically better output for complex documents.

4. **Artifacts compound**: Uncleaned artifacts (orphaned captions, page numbers) propagate through chunking and embedding, degrading retrieval quality.

5. **Regex patterns are corpus-specific**: The cleaning patterns here target neuroscience textbook artifacts (LO markers, figure captions). Other corpora may need different patterns.

---

## References

- [Docling GitHub](https://github.com/docling-project/docling)
- [IBM Docling Announcement](https://research.ibm.com/blog/docling-generative-AI)
- [Docling AAAI 2025 Paper](https://research.ibm.com/publications/docling-an-efficient-open-source-toolkit-for-ai-driven-document-conversion)
- [PyMuPDF4LLM Documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- [PyMuPDF Layout](https://pymupdf.readthedocs.io/en/latest/pymupdf-layout/index.html)
