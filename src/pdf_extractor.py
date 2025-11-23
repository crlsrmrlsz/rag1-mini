"""Extract text from PDFs to Markdown for RAG pipeline.

This module handles Phase 1 of the RAG1-Mini project: PDF text extraction.
It processes PDFs from data/raw/, preserving folder structure in data/processed/.
"""

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


class PDFExtractor:
    """Extract text from PDFs to Markdown format."""
    
    def __init__(self, raw_dir: Path, processed_dir: Path):
        """Initialize extractor with source and destination directories.
        
        Args:
            raw_dir: Directory containing raw PDF files
            processed_dir: Directory for processed Markdown files
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        
    def find_pdfs(self) -> List[Path]:
        """Find all PDFs in raw directory and immediate subdirectories.
        
        Returns:
            List of PDF file paths
        """
        pdfs = []
        
        # Root level PDFs
        pdfs.extend(self.raw_dir.glob("*.pdf"))
        
        # One level of subdirectories
        for subdir in self.raw_dir.iterdir():
            if subdir.is_dir():
                pdfs.extend(subdir.glob("*.pdf"))
        
        return sorted(pdfs)
    
    def get_output_path(self, pdf_path: Path) -> Path:
        """Calculate output path maintaining folder structure.
        
        Args:
            pdf_path: Source PDF file path
            
        Returns:
            Destination Markdown file path
        """
        # Get relative path from raw_dir
        relative_path = pdf_path.relative_to(self.raw_dir)
        
        # Change extension to .md
        md_filename = relative_path.with_suffix('.md')
        
        # Build full output path
        return self.processed_dir / md_filename
    
    def extract_single_pdf(self, pdf_path: Path) -> bool:
        """Extract text from a single PDF to Markdown.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if extraction succeeded, False otherwise
        """
        try:
            logger.info(f"Processing: {pdf_path.name}")
            
            # Extract markdown text without headers and footers
            md_text = pymupdf4llm.to_markdown(
                str(pdf_path),
                header=False,
                footer=False
            )
            
            # Calculate output path
            output_path = self.get_output_path(pdf_path)
            
            # Create parent directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write markdown file
            output_path.write_bytes(md_text.encode())
            
            logger.info(f"Saved: {output_path.relative_to(self.processed_dir)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            return False
    
    def extract_all(self) -> None:
        """Extract text from all PDFs in raw directory."""
        pdfs = self.find_pdfs()
        
        if not pdfs:
            logger.warning(f"No PDFs found in {self.raw_dir}")
            return
        
        logger.info(f"Found {len(pdfs)} PDF(s) to process")
        
        successful = 0
        failed = 0
        
        for pdf_path in pdfs:
            if self.extract_single_pdf(pdf_path):
                successful += 1
            else:
                failed += 1
        
        logger.info(
            f"Extraction complete: {successful} succeeded, {failed} failed"
        )


def main():
    """Main entry point for PDF extraction."""
    # Define project directories
    project_root = Path(__file__).parent.parent  # Go up to project root
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    
    # Validate raw directory exists
    if not raw_dir.exists():
        logger.error(f"Raw directory not found: {raw_dir}")
        return
    
    # Extract PDFs
    extractor = PDFExtractor(raw_dir, processed_dir)
    extractor.extract_all()


if __name__ == "__main__":
    main()