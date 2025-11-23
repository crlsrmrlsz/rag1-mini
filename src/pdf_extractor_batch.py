"""Batch PDF extraction script with progress tracking.

This script processes PDFs in smaller batches to avoid timeout issues.
"""

import logging
from pathlib import Path
from typing import List
import time

import pymupdf.layout  # Must be imported first to activate layout features
import pymupdf4llm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_all_pdfs(raw_dir: Path) -> List[Path]:
    """Find all PDFs in raw directory and immediate subdirectories."""
    pdfs = []
    
    # Root level PDFs
    pdfs.extend(raw_dir.glob("*.pdf"))
    
    # One level of subdirectories
    for subdir in raw_dir.iterdir():
        if subdir.is_dir():
            pdfs.extend(subdir.glob("*.pdf"))
    
    return sorted(pdfs)


def get_output_path(raw_dir: Path, processed_dir: Path, pdf_path: Path) -> Path:
    """Calculate output path maintaining folder structure."""
    # Get relative path from raw_dir
    relative_path = pdf_path.relative_to(raw_dir)
    
    # Change extension to .md
    md_filename = relative_path.with_suffix('.md')
    
    # Build full output path
    return processed_dir / md_filename


def extract_single_pdf(raw_dir: Path, processed_dir: Path, pdf_path: Path) -> bool:
    """Extract text from a single PDF to Markdown."""
    try:
        logger.info(f"Processing: {pdf_path.relative_to(raw_dir)}")
        
        # Extract markdown text without headers and footers
        start_time = time.time()
        md_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            header=False,
            footer=False
        )
        extraction_time = time.time() - start_time
        
        # Calculate output path
        output_path = get_output_path(raw_dir, processed_dir, pdf_path)
        
        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write markdown file
        output_path.write_bytes(md_text.encode())
        
        file_size = len(md_text)
        logger.info(f"Saved: {output_path.relative_to(processed_dir)} ({file_size} chars, {extraction_time:.1f}s)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {pdf_path.relative_to(raw_dir)}: {e}")
        return False


def process_pdf_batch(raw_dir: Path, processed_dir: Path, pdfs: List[Path], batch_num: int, batch_size: int = 3):
    """Process a batch of PDFs."""
    start_idx = (batch_num - 1) * batch_size
    end_idx = min(start_idx + batch_size, len(pdfs))
    batch = pdfs[start_idx:end_idx]
    
    logger.info(f"Processing batch {batch_num}: files {start_idx + 1}-{end_idx} of {len(pdfs)}")
    
    successful = 0
    failed = 0
    
    for pdf_path in batch:
        if extract_single_pdf(raw_dir, processed_dir, pdf_path):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Batch {batch_num} complete: {successful} succeeded, {failed} failed")
    return successful, failed


def main():
    """Main entry point for batch PDF extraction."""
    # Define project directories
    project_root = Path(__file__).parent.parent  # Go up to project root
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    
    # Validate raw directory exists
    if not raw_dir.exists():
        logger.error(f"Raw directory not found: {raw_dir}")
        return
    
    # Find all PDFs
    pdfs = find_all_pdfs(raw_dir)
    
    if not pdfs:
        logger.warning(f"No PDFs found in {raw_dir}")
        return
    
    logger.info(f"Found {len(pdfs)} PDF(s) to process")
    
    # Process in batches of 3 to avoid timeouts
    batch_size = 3
    total_batches = (len(pdfs) + batch_size - 1) // batch_size
    
    total_successful = 0
    total_failed = 0
    
    for batch_num in range(1, total_batches + 1):
        successful, failed = process_pdf_batch(raw_dir, processed_dir, pdfs, batch_num, batch_size)
        total_successful += successful
        total_failed += failed
        
        # Small delay between batches
        if batch_num < total_batches:
            logger.info("Waiting 5 seconds before next batch...")
            time.sleep(5)
    
    logger.info(f"Extraction complete: {total_successful} succeeded, {total_failed} failed")


if __name__ == "__main__":
    main()