# PDF Extraction

> **Library:** [Docling](https://github.com/DS4SD/docling) | IBM Research

*Documentation pending.*

## TL;DR

Converts PDF documents to structured Markdown, preserving headings, paragraphs, and document structure for downstream chunking.

## Topics to Cover

- Why Docling over PyMuPDF, pdfplumber, or other libraries
- Handling complex layouts (multi-column, figures, tables)
- Scientific book challenges (footnotes, equations, bibliographies)
- Post-processing and cleaning requirements
- Performance characteristics

## Key Files

- `src/content_preparation/extraction/docling_parser.py`
- `src/content_preparation/cleaning/text_cleaner.py`

## Related

- [Section Chunking](../chunking/section-chunking.md) — Uses Markdown structure from extraction
