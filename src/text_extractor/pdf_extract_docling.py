import logging
from pathlib import Path
from typing import List, Iterable

# Updated imports to support OCR configuration
from docling.datamodel.document import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

class BatchPDFConverter:
    """
    Handles batch conversion of PDFs to Markdown while preserving
    directory structure. Configured to skip OCR for digital PDFs.
    """

    def __init__(self, raw_dir: Path, processed_dir: Path):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        
        # ---- CONFIGURATION CHANGES START HERE ----
        
        # 1. Define Pipeline Options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False           # <--- Force OCR OFF (Fast & clean for digital docs)
        pipeline_options.do_table_structure = True # Keep table recognition ON
        
        # 2. Initialize Converter with specific PDF options
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        # ---- CONFIGURATION CHANGES END HERE ----

    def _get_pdf_paths(self) -> List[Path]:
        """Recursively finds all .pdf files in the raw directory."""
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.raw_dir}")
        
        pdfs = list(self.raw_dir.rglob("*.pdf"))
        logger.info(f"Found {len(pdfs)} PDF files in {self.raw_dir}")
        return pdfs

    def _get_output_path(self, source_path: Path) -> Path:
        """
        Determines the mirror output path for a given source PDF.
        Example: data/raw/folder/doc.pdf -> data/processed/folder/doc.md
        """
        relative_path = source_path.relative_to(self.raw_dir)
        destination = self.processed_dir / relative_path.with_suffix(".md")
        return destination

    def run(self):
        """Executes the batch conversion process."""
        pdf_paths = self._get_pdf_paths()

        if not pdf_paths:
            logger.warning("No PDFs found to process.")
            return

        logger.info(f"Starting batch conversion of {len(pdf_paths)} files... (OCR Disabled)")
        
        # docling.convert_all handles the iteration efficiently
        conversion_results = self.converter.convert_all(
            pdf_paths,
            raises_on_error=False # Continue even if one file fails
        )

        self._save_results(conversion_results)

    def _save_results(self, results: Iterable):
        """Iterates through conversion results and saves them to disk."""
        success_count = 0
        failure_count = 0

        for result in results:
            source_path = result.input.file 
            output_path = self._get_output_path(source_path)

            if result.status == ConversionStatus.SUCCESS:
                try:
                    # 1. Ensure destination directory exists
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 2. Export to Markdown
                    markdown_content = result.document.export_to_markdown()
                    
                    # 3. Write file
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(markdown_content)
                    
                    logger.info(f"✔ Converted: {source_path.name} -> {output_path}")
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"✘ Error saving {source_path.name}: {e}")
                    failure_count += 1
            else:
                logger.error(f"✘ Conversion failed for {source_path.name}. Errors: {result.errors}")
                failure_count += 1

        logger.info("-" * 40)
        logger.info(f"Processing Complete. Success: {success_count} | Failed: {failure_count}")

# ---- Entry Point ----
if __name__ == "__main__":
    # Define Project Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

    # Run
    processor = BatchPDFConverter(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)
    processor.run()