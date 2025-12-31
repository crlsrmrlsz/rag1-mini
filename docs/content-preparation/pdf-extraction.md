# PDF Extraction

> **Library:** [Docling](https://github.com/DS4SD/docling) | IBM Research
> **Stage:** 1 of 8 | Converts PDFs to structured Markdown

Docling provides layout-aware PDF parsing that preserves document structure (headings, paragraphs, sections) for downstream chunking. This stage removes non-content elements (figures, tables, footnotes) and exports clean Markdown.

## TL;DR

```bash
python -m src.stages.run_stage_1_extraction
```

Converts all PDFs in `data/raw/` to Markdown files in `data/processed/01_raw_extraction/`. Uses Docling's layout detection to preserve document structure while removing artifacts.

## Why Docling?

| Library | Layout Detection | Markdown Export | OCR | Tables | Notes |
|---------|-----------------|-----------------|-----|--------|-------|
| **Docling** | Yes | Native | Optional | Optional | IBM Research, structured output |
| PyMuPDF | No | Manual | Via Tesseract | Limited | Fast, low-level extraction |
| pdfplumber | Limited | Manual | No | Yes | Table-focused |
| unstructured.io | Yes | Native | Yes | Yes | Heavy dependencies, API option |

**Choice rationale:**
- Native Markdown export with heading hierarchy (H1, H2, etc.)
- Layout detection handles multi-column and complex scientific layouts
- Clean API for removing document elements (figures, footnotes)
- No OCR dependency (our PDFs are text-based)
- Reasonable performance for book-length documents

## Implementation

### Core Function

```python
# src/content_preparation/extraction/docling_parser.py

def extract_pdf(pdf_path: Path) -> str:
    """Extract text from PDF to markdown.

    Removes captions, footnotes, tables, page headers/footers,
    and pictures with their children.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        Extracted text as markdown string.
    """
    converter = _get_converter()
    result = converter.convert(pdf_path)
    doc = result.document

    # Remove artifacts
    labels_to_remove = {
        DocItemLabel.CAPTION,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.TABLE
    }
    # ... filtering logic

    return doc.export_to_markdown()
```

### Configuration

```python
# Docling pipeline options
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False           # Skip OCR (text PDFs only)
pipeline_options.do_table_structure = False  # Skip table parsing
```

### Removed Elements

| Element | Why Removed |
|---------|-------------|
| `CAPTION` | Figure captions without figures are noise |
| `FOOTNOTE` | Disrupts text flow, often references |
| `PAGE_HEADER` | Running headers with chapter titles |
| `PAGE_FOOTER` | Page numbers, publisher info |
| `TABLE` | Complex structure, separate handling needed |
| `PICTURE` | Figures don't translate to text |

## Scientific Book Challenges

Our corpus (neuroscience + philosophy) presents specific extraction challenges:

### Multi-Column Layouts
Textbooks often use two columns. Docling handles this via layout detection, merging columns in reading order.

### Footnotes and Endnotes
Scientific texts have extensive footnotes. We remove them during extraction to avoid fragmenting the main argument. Future enhancement: extract as metadata.

### Mathematical Equations
Inline math and equations often extract as garbled Unicode. Current approach: let the LLM interpret during chunking context.

### Bibliographies
Reference sections extract as dense text. The section chunker uses heading detection to isolate these.

### Box Insets
Highlighted "key concepts" boxes may extract out of order. Manual review stage catches major issues.

## Post-Processing: Cleaning Stage

After extraction, `run_stage_2_processing` applies regex-based cleaning:

```python
# src/content_preparation/cleaning/text_cleaner.py

# Configurable patterns in src/config.py:
LINE_REMOVAL_PATTERNS = [
    ("page_numbers", r"^\s*\d{1,4}\s*$"),
    ("chapter_markers", r"^Chapter\s+\d+"),
    # ... more patterns
]

INLINE_REMOVAL_PATTERNS = [
    ("citation_numbers", r"\[\d+(?:,\s*\d+)*\]"),
    # ... more patterns
]
```

**Cleaning outputs:**
- Cleaned Markdown: `data/processed/03_markdown_cleaning/`
- Cleaning report: Documents all removals for audit

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Speed | ~30 sec/book | Depends on PDF complexity |
| Memory | ~500MB peak | Per-book processing |
| Output size | ~60% of input | After artifact removal |

## Troubleshooting

### "No text extracted"
- Check if PDF is image-based (scanned). Enable `do_ocr = True`.
- Verify PDF isn't encrypted or malformed.

### Garbled Unicode
- Common with mathematical notation. Accept or fix in manual review.
- Some fonts use private-use Unicode characters.

### Wrong heading hierarchy
- Docling infers headings from font size/weight. Manual review may need fixes.
- Use regex in cleaning stage for consistent chapter markers.

### Missing sections
- Check if content is in image-only blocks.
- Tables removed by default—may contain important text.

## Pipeline Flow

```
data/raw/*.pdf
     │
     ▼ Stage 1: extract_pdf()
data/processed/01_raw_extraction/*.md
     │
     ▼ Manual review (optional)
data/processed/02_manual_review/*.md
     │
     ▼ Stage 2: run_structural_cleaning()
data/processed/03_markdown_cleaning/*.md
     │
     ▼ Stage 3: segment_document()
data/processed/04_nlp_chunks/*.json
```

## Key Files

| File | Purpose |
|------|---------|
| `src/content_preparation/extraction/docling_parser.py` | Core extraction function |
| `src/content_preparation/cleaning/text_cleaner.py` | Regex-based post-processing |
| `src/stages/run_stage_1_extraction.py` | CLI runner for batch extraction |
| `src/config.py` | Cleaning patterns configuration |

## Related

- [Section Chunking](../chunking/section-chunking.md) — Uses Markdown structure from extraction
- [Architecture](../architecture.md) — Full 8-stage pipeline overview
