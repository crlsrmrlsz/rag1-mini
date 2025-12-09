from pathlib import Path

from .config import DATA_DIR, DIR_RAW_EXTRACT
from .utils import setup_logging, get_file_list, get_output_path
from .extractors import DoclingExtractor

logger = setup_logging("Stage1_Extraction")

def main():
    raw_dir = DATA_DIR / "raw"

    # 1. Initialize Extractor
    logger.info("Initializing Docling Extractor (No OCR, No Tables)...")
    extractor = DoclingExtractor()

    # 2. Find Files
    pdf_files = get_file_list(raw_dir, "pdf")
    logger.info(f"Found {len(pdf_files)} PDFs in {raw_dir}")

    if not pdf_files:
        return

    # 3. Batch Process
    success_count = 0
    for pdf_path in pdf_files:
        try:
            # Calculate output path (Mirroring structure)
            output_path = get_output_path(pdf_path, raw_dir, DIR_RAW_EXTRACT, ".md")

            # Skip if already exists (optional, good for resuming)
            if output_path.exists():
                logger.info(f"Skipping {pdf_path.name} (Output exists)")
                continue

            logger.info(f"Converting: {pdf_path.name}")

            # Execute Extraction
            md_content = extractor.convert_pdf(pdf_path)

            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(md_content, encoding="utf-8")

            success_count += 1
            logger.info(f"✔ Saved to: {output_path}")

        except Exception as e:
            logger.error(f"✘ Failed {pdf_path.name}: {e}")

    logger.info(f"Stage 1 Complete. {success_count}/{len(pdf_files)} files processed.")
    logger.info(f"NEXT STEP: Review Markdown files in {DIR_RAW_EXTRACT}, then move valid files to {DATA_DIR / 'processed' / '02_manual_review'}")

if __name__ == "__main__":
    main()
