# Content Preparation

> **Stages:** 1-3 of 8 | Converts PDFs to clean, segmented Markdown ready for chunking

This document covers the complete workflow for preparing content from PDF books to generate clean markdown files ready to chunk.

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

Before any automated extraction, PDFs are manually cleaned using PDF editing tools to remove the pages that contain these elements:


| Element | Why Remove |
|---------|------------|
| References/Bibliography | Dense citation blocks confuse layout detection |
| Glossary | Alphabetical lists extract poorly |
| Index | Page number lists are noise |
| Acknowledgments | Not core content |
| Notes sections | Often formatted as footnotes, sometimes at the end of chapters or end of book |
| Appendices | Supplementary material, separate handling needed |

Pre-cleaning reduces the complexity for the next phases and contributes to the quality of the data downstream. I included this initial manual cleaning after realizing the complexity of getting clean markdown for scientific-style books, but for a bigger project this won't scale. With expected future improvements in PDF text extractor models, this cleaning could be done during text extraction itself or afterwards over a properly structured markdown, removing unnecessary sections.

This phase was done with [PDF24](https://www.pdf24.org/).



## Phase 2: PDF to Markdown Conversion

This phase took considerably more effort than expected.

First attempts used [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/), a lightweight PDF-to-markdown converter. For scientific books, however, it does not work well with multicolumn layouts, especially when images appear mid-paragraph. The [PyMuPDF-Layout](https://pymupdf.readthedocs.io/en/latest/pymupdf-layout/index.html) extension adds AI-based layout detection, which improves results significantly but still not enough. 


### Solution: Docling

[Docling](https://github.com/docling-project/docling) (IBM Research) uses AI vision models for layout understanding, solving many of the problems PyMuPDF4LLM couldn't handle. I didn't need tables or images, which are possibly the most difficult elements to extract, so I couldn't test it thoroughly.

Most cases worked well with multicolumn layouts, but edge cases like this one still failed, mixing columns incorrectly.

 ![Multi column page](../../assets/page_columns.png)

It allows you to directly remove some elements like captions, tables, figures, headers or footer that simplify next cleaning phase.


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


## Phase 3: Markdown Cleaning

After Docling extraction, two cleaning steps refine the output.

### Manual Review (Optional)

Location: `data/processed/02_manual_review/`

Purpose: Catch extraction errors before automated cleaning:
- Solve multicolumn errors (more than 40 pages with this problem, lot of manual work, not scalable)
- Verify heading hierarchy (all headers were second level markdown header, did not get proper hierarchy)
- Fix obvious extraction failures
- Remove any remaining artifacts

### Automated Cleaning

After manual inspection, I identified common patterns suitable for automated cleaning using regex. Each book had different specific errors from conversion, so this was again a very manual, non-scalable task. 

The automated cleaner applies regex patterns to remove common artifacts.

#### Line Removal Patterns

Full lines matching these patterns are deleted:

| Pattern | Example Match | Purpose |
|---------|---------------|---------|
| `FIGURE_TABLE_CAPTION` | "Figure 2. Model diagram" | Remove orphaned captions |
| `LEARNING_OBJECTIVE` | "LO 1.2" | Remove textbook learning objectives |
| `SINGLE_CHAR` | "a" (isolated line) | Remove OCR noise, diagram labels |
| `HEADING_SINGLE_NUMBER` | "## 5" | Remove meaningless number-only headings |


#### Inline Removal Patterns

Text within lines matching these patterns is removed:

| Pattern | Example | Purpose |
|---------|---------|---------|
| `FIG_TABLE_REF` | "(Figure 2)" | Remove parenthetical figure references |
| `FOOTNOTE_MARKER` | "fn3" | Remove footnote markers mid-sentence |
| `TRAILING_NUMBER` | ". 81 We" → ". We" | Remove page numbers between sentences |



#### Character Substitutions

| Original | Replacement | Purpose |
|----------|-------------|---------|
| `\u2014` | `--` | Unicode em-dash escape sequence |
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
