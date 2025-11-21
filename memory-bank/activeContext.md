# Active Context - Phase 1: PDF Extraction

## Current Focus

Testing and comparing PDF text extraction methods to solve text ordering issues in multi-column academic documents (neuroscience and philosophy texts).

## Extraction Methods Being Tested

### 1. pdf_extract_pymupdf_dict.py (Basic)
- **Approach**: Uses `page.get_text("dict")` without sorting
- **Status**: ❌ Wrong column order (right column first, then left)
- **Debug Color**: Red boundaries

### 2. pdf_extract_pymupdf_dict_sorted.py (Sorted)
- **Approach**: Uses `page.get_text("dict", sort=True)`
- **Status**: ⚠️ Partial improvement but still mixes columns based on Y-position
- **Debug Color**: Blue boundaries

### 3. pdf_extract_pymupdf_blocks.py (K-means Clustering)
- **Approach**: K-means clustering for column detection
- **Status**: ✅ Usually correct column order but ⚠️ misses some text blocks
- **Debug Color**: Red boundaries
- **Best for**: Column handling, but incomplete coverage

### 4. pdf_extract_pymupdf4llm.py (RAG-optimized)
- **Approach**: Uses pymupdf4llm library optimized for RAG
- **Status**: ✅ Debug visualization working, extensive text extraction
- **Debug Color**: Orange boundaries
- **Issue**: Processing timeout on large documents
- **Next**: Test on smaller document segments

## Debug Visualization

Each method generates debug PDFs (pages 1-5) with color-coded boundaries:
- **Location**: `data/debug/{method_name}/page_{number}.pdf`
- **Purpose**: Visual inspection of text ordering and block coverage

## Recent Changes

- ✅ Removed `pdf_extract_pymupdf_multicolumn.py` (flawed manual approach)
- ✅ Fixed PyMuPDF4LLM debug visualization
- ✅ Simplified README.md and memory-bank structure

## Next Steps

1. **Visual Assessment**: Manually review debug PDFs to compare methods
2. **Method Selection**: Choose optimal approach based on:
   - Text ordering quality (left-to-right, top-to-bottom)
   - Column handling accuracy
   - Block coverage completeness
3. **Production Implementation**: Integrate selected method into ingestion pipeline
4. **Move to Phase 2**: Begin chunking and embedding once extraction is solid

## Key Decisions

- **Quality over speed**: Taking time to get extraction right before moving forward
- **Visual validation**: Debug PDFs are primary quality assessment tool
- **Systematic comparison**: Testing multiple approaches to find best solution
