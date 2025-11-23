"""Single PDF extraction script for controlled processing."""

import logging
from pathlib import Path
from typing import List

import pymupdf.layout  # Must be imported first to activate layout features
import pymupdf4llm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_remaining_pdfs(raw_dir: Path, processed_dir: Path) -> List[Path]:
    """Find PDFs that haven't been processed yet."""
    # Find all PDFs
    all_pdfs = []
    
    # Root level PDFs
    all_pdfs.extend(raw_dir.glob("*.pdf"))
    
    # One level of subdirectories
    for subdir in raw_dir.iterdir():
        if subdir.is_dir():
            all_pdfs.extend(subdir.glob("*.pdf"))
    
    # Filter out already processed PDFs
    remaining_pdfs = []
    for pdf_path in all_pdfs:
        # Calculate expected output path
        relative_path = pdf_path.relative_to(raw_dir)
        expected_output = processed_dir / relative_path.with_suffix('.md')
        
        if not expected_output.exists():
            remaining_pdfs.append(pdf_path)
    
    return sorted(remaining_pdfs)


def get_output_path(raw_dir: Path, processed_dir: Path, pdf_path: Path) -> Path:
    """Calculate output path maintaining folder structure."""
    relative_path = pdf_path.relative_to(raw_dir)
    md_filename = relative_path.with_suffix('.md')
    return processed_dir / md_filename


def extract_single_pdf(raw_dir: Path, processed_dir: Path, pdf_path: Path) -> bool:
    """Extract text from a single PDF to Markdown."""
    try:
        logger.info(f"Processing: {pdf_path.relative_to(raw_dir)}")
        
        # Extract markdown text without headers and footers
        md_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            header=False,
            footer=False
        )
        
        # Calculate output path
        output_path = get_output_path(raw_dir, processed_dir, pdf_path)
        
        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write markdown file
        output_path.write_bytes(md_text.encode())
        
        file_size = len(md_text)
        logger.info(f"✓ Saved: {output_path.relative_to(processed_dir)} ({file_size:,} chars)")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to process {pdf_path.relative_to(raw_dir)}: {e}")
        return False


def main():
    """Main entry point for single PDF extraction."""
    # Define project directories
    project_root = Path(__file__).parent.parent  # Go up to project root
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    
    # Validate raw directory exists
    if not raw_dir.exists():
        logger.error(f"Raw directory not found: {raw_dir}")
        return
    
    # Find remaining PDFs
    remaining_pdfs = find_remaining_pdfs(raw_dir, processed_dir)
    
    if not remaining_pdfs:
        logger.info("All PDFs have been processed!")
        return
    
    logger.info(f"Found {len(remaining_pdfs)} PDF(s) to process")
    
    # Process remaining PDFs
    for i, pdf_path in enumerate(remaining_pdfs, 1):
        logger.info(f"--- Processing {i}/{len(remaining_pdfs)} ---")
        extract_single_pdf(raw_dir, processed_dir, pdf_path)


if __name__ == "__main__":
    main()