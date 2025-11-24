"""Extract text from PDFs to Markdown for a RAG pipeline.

This module handles Phase 1 of the RAG1-Mini project: PDF text extraction.
It recursively processes PDFs from data/raw/, preserving folder structure
in data/processed/.
"""

import logging
from pathlib import Path
from typing import List

import pymupdf.layout  # Required import to enable PyMuPDF layout features
import pymupdf4llm     # High-level Markdown extraction utilities


# Configure global logging for the module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text from PDFs and export Markdown files preserving directory layout."""
    
    def __init__(self, raw_dir: Path, processed_dir: Path):
        """
        Args:
            raw_dir: Root directory containing input PDF files.
            processed_dir: Output directory where Markdown files will be written.
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        
    def find_pdfs(self) -> List[Path]:
        """Recursively find all PDFs under raw_dir.
        
        Returns:
            A sorted list of PDF file paths.
        """
        # rglob allows unlimited-depth recursive search for *.pdf
        return sorted(self.raw_dir.rglob("*.pdf"))
    
    def get_output_path(self, pdf_path: Path) -> Path:
        """Compute the corresponding Markdown output path for a PDF.
        
        Preserves folder structure relative to raw_dir.
        
        Args:
            pdf_path: Path to the source PDF file.
            
        Returns:
            Path where the Markdown file should be stored.
        """
        # Determine the relative path (subfolders included)
        relative_path = pdf_path.relative_to(self.raw_dir)
        
        # Swap file extension to .md while keeping directory structure
        md_filename = relative_path.with_suffix('.md')
        
        # Place file inside processed_dir
        return self.processed_dir / md_filename
    def extract_single_pdf_debug(self, pdf_path: Path) -> bool:
        """Extract a single PDF into Markdown format.
        
        Args:
            pdf_path: Path of the PDF to extract.
            
        Returns:
            True if processing succeeded, False otherwise.
        """

        out_path = self.get_output_path(pdf_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            md = pymupdf4llm.to_markdown(str(pdf))
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md)
        except Exception as e:
            logging.exception(f"Failed to process {pdf}")  # <-- ESTA LÍNEA DA TRACEBACK COMPLETO


        
    def extract_single_pdf(self, pdf_path: Path) -> bool:
        """Extract a single PDF into Markdown format.
        
        Args:
            pdf_path: Path of the PDF to extract.
            
        Returns:
            True if processing succeeded, False otherwise.
        """
        try:
            logger.info(f"Processing: {pdf_path}")
            
            # Extract Markdown using PyMuPDF4LLM.

            # Compute destination path and ensure the folder exists
            output_path = self.get_output_path(pdf_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            md = pymupdf4llm.to_markdown(str(pdf_path))
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md)

            
        except Exception as e:
            logging.exception(f"Failed to process {pdf_path}")  # <-- ESTA LÍNEA DA TRACEBACK COMPLETO
    
    def extract_all(self) -> None:
        """Extract all PDF files found under raw_dir."""
        pdfs = self.find_pdfs()
        # TEMP: list of PDFs to test
        test_only = {
            "Cognitive_Biology_Evolutionary_and_Developmental_Perspectives_on_Mind_Brain_and_Behavior_Luca_Tommasi_Mary_A._Peterson_Lynn_Nadel.pdf",
            "Brain_and_behavior_a_cognitive_neuroscience_perspective_David_Eagleman_Jonathan_Downar.pdf",
            "Biopsychology_Global_Ed_11th_John_Pinel_Steven_Barnes.pdf",
        }

        # Filter PDFs to only those in test list
        pdfs = [p for p in pdfs if p.name in test_only]

        if not pdfs:
            logger.warning(f"No PDFs found in {self.raw_dir}")
            return
        
        logger.info(f"Found {len(pdfs)} PDF(s) to process")
        
        successful = 0
        failed = 0
        
        # Sequential processing to keep logs readable and stable
        for pdf_path in pdfs:
            if self.extract_single_pdf(pdf_path):
                successful += 1
            else:
                failed += 1
        
        logger.info(
            f"Extraction complete: {successful} succeeded, {failed} failed"
        )


def main():
    """Define project paths and run the extraction pipeline."""
    
    # Project root is two levels above this file (adjust if needed)
    project_root = Path(__file__).parent.parent
    
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    
    # Ensure input directory exists before starting
    if not raw_dir.exists():
        logger.error(f"Raw directory not found: {raw_dir}")
        return
    
    extractor = PDFExtractor(raw_dir, processed_dir)
    extractor.extract_all()


if __name__ == "__main__":
    main()
