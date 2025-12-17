"""Stage 1: Extract PDF files to markdown using Docling."""

from pathlib import Path

from src.config import DATA_DIR, DIR_RAW_EXTRACT
from src.utils import setup_logging, get_file_list, get_output_path
from src.extractors import extract_pdf

logger = setup_logging("Stage1_Extraction")


def main():
    """Run PDF extraction pipeline."""
    raw_dir = DATA_DIR / "raw"

    logger.info("Starting Stage 1: PDF Extraction")

    # Find PDF files
    pdf_files = get_file_list(raw_dir, "pdf")
    logger.info(f"Found {len(pdf_files)} PDFs in {raw_dir}")

    if not pdf_files:
        logger.warning("No PDF files found. Exiting.")
        return

    # Process each PDF
    success_count = 0
    for pdf_path in pdf_files:
        output_path = get_output_path(pdf_path, raw_dir, DIR_RAW_EXTRACT, ".md")

        # Skip if already exists
        if output_path.exists():
            logger.info(f"Skipping {pdf_path.name} (output exists)")
            continue

        logger.info(f"Extracting: {pdf_path.name}")

        md_content = extract_pdf(pdf_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md_content, encoding="utf-8")

        success_count += 1
        logger.info(f"Saved: {output_path}")

    logger.info(f"Stage 1 complete. {success_count}/{len(pdf_files)} files processed.")


if __name__ == "__main__":
    main()
