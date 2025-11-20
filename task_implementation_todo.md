# PDF Extraction Improvement Implementation

## Task: Create Improved Dict-based PDF Extractor with Sorting

### Completed Items
- [x] Create organized pdf_extractors folder structure
- [x] Move existing extractors to new organized location  
- [x] Implement new dict_sorted extractor with page.get_text("dict", sort=True)
- [x] Set up proper debug folder structure (data/debug/pdf_extract_pymupdf_dict_sorted/)
- [x] Create module __init__.py for clean imports
- [x] Use blue color for debug visualization (distinct from other methods)
- [x] Maintain consistent interface with other extraction methods
- [x] Include example usage and documentation

### Files Created/Modified
1. **src/pdf_extractors/__init__.py** - Module initialization and imports
2. **src/pdf_extractors/pdf_extract_pymupdf_dict_sorted.py** - NEW sorted dict extractor
3. **src/pdf_extractors/pdf_extract_pymupdf_dict.py** - Moved from src/
4. **src/pdf_extractors/pdf_extract_pymupdf_blocks.py** - Moved from src/

### Implementation Details
- **Key Change**: Uses `page.get_text("dict", sort=True)` instead of `page.get_text("dict")`
- **Debug Output**: Generates PDF visualizations to `data/debug/pdf_extract_pymupdf_dict_sorted/`
- **Color Coding**: Blue boundaries for sorted method vs red for basic dict method
- **Consistency**: Same interface as other methods for easy comparison
- **Benefits**: Should improve text ordering issues in multi-column academic documents

### Usage
```python
from src.pdf_extractors.pdf_extract_pymupdf_dict_sorted import extract_document_sorted_with_debug

result = extract_document_sorted_with_debug(
    'data/raw/ch1_ch14_Brain_and_behavior.pdf',
    'data/debug/pdf_extract_pymupdf_dict_sorted'
)
```

### Next Steps
- Test implementation in active conda environment
- Compare text ordering quality vs basic dict method
- Evaluate extraction performance and accuracy
- Update memory bank documentation to reflect new method
