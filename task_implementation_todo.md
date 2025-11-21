# Phase 1: PDF Extraction - Current Tasks

## Objective
Select the optimal PDF text extraction method for multi-column academic documents (neuroscience and philosophy texts).

## Available Methods

### 1. pdf_extract_pymupdf_dict.py
- Basic dict extraction
- ❌ Wrong column order (right then left)

### 2. pdf_extract_pymupdf_dict_sorted.py
- Sorted dict extraction
- ⚠️ Partial improvement, still mixes columns

### 3. pdf_extract_pymupdf_blocks.py
- K-means clustering for columns
- ✅ Good column order
- ⚠️ Misses some text blocks

### 4. pdf_extract_pymupdf4llm.py
- RAG-optimized library
- ✅ Extensive extraction
- ⚠️ Timeout on large documents

## Current Tasks

### Immediate (This Week)
- [ ] Visual assessment of debug PDFs for all 4 methods
- [ ] Compare text ordering quality across methods
- [ ] Evaluate block coverage completeness
- [ ] Select optimal method for production use

### Next Steps
- [ ] Integrate selected method into `ingest.py`
- [ ] Process full document set (Eagleman + philosophy texts)
- [ ] Generate `clean_paragraphs.jsonl` output
- [ ] Manual validation of extraction quality

## Evaluation Criteria

1. **Text Ordering**: Proper left-to-right, top-to-bottom reading flow
2. **Column Handling**: Accurate multi-column detection and processing
3. **Block Coverage**: Minimal missing content
4. **Reliability**: Consistent results across different page layouts

## Debug Output Location

```
data/debug/
├── pdf_extract_pymupdf_dict/
├── pdf_extract_pymupdf_dict_sorted/
├── pdf_extract_pymupdf_blocks/
└── pdf_extract_pymupdf4llm/
```

Each contains debug PDFs (pages 1-5) with color-coded text block boundaries.

## Success Criteria for Phase 1

- ✅ Method selected based on visual quality assessment
- ✅ Clean text extraction with proper reading order
- ✅ Multi-column layouts handled correctly
- ✅ Output format: `{text, page}` in JSONL

## Notes

- Removed `pdf_extract_pymupdf_multicolumn.py` (flawed manual approach)
- Focus on quality over speed - get extraction right before moving to Phase 2
- Visual validation is primary quality metric
