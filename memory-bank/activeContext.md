# Active Context - Phase 2: Manual Review & Processing

## Current State
Phase 1 (PDF Extraction) is complete. The extracted Markdown files are located in `data/processed/01_raw_extraction/`.

The current task is to manually review these files for any extraction errors.

## Next Steps

1.  **Manual Review**: Go through the files in `data/processed/01_raw_extraction/`, clean up any issues, and move the corrected files to `data/processed/02_manual_review/`.
2.  **Run Stage 2**: Execute `python src/run_stage_2_processing.py` to process the reviewed files into structured chunks with metadata.

The chunking logic has been updated to include the book name for better context.
