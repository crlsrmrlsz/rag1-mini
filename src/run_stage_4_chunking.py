"""Stage 4: Section-aware chunking with token limits and overlap."""

from pathlib import Path

from src.config import DIR_NLP_CHUNKS, DIR_FINAL_CHUNKS
from src.utils import setup_logging, get_file_list
from src.ingest import run_section_chunking

logger = setup_logging("Stage4_Chunking")


def main():
    """Run section chunking pipeline."""
    logger.info("Starting Stage 4: Section Chunking")

    # Check Stage 3 output exists
    nlp_chunk_files = get_file_list(DIR_NLP_CHUNKS, "json")
    logger.info(f"Found {len(nlp_chunk_files)} NLP chunk files from Stage 3.")

    if not nlp_chunk_files:
        logger.warning(f"No NLP chunk files found in {DIR_NLP_CHUNKS}. Run Stage 3 first.")
        return

    # Run section chunking
    stats = run_section_chunking()

    # Verify output
    section_dir = DIR_FINAL_CHUNKS / "section"
    section_files = list(section_dir.glob("*.json")) if section_dir.exists() else []

    logger.info(f"Stage 4 complete. {len(section_files)} files created.")
    logger.info(f"Total chunks: {sum(stats.values())}")
    logger.info(f"Output: {section_dir}")


if __name__ == "__main__":
    main()
