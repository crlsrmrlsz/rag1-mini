from pathlib import Path

from src.config import DIR_NLP_CHUNKS, DIR_FINAL_CHUNKS
from src.utils.file_utils import setup_logging, get_file_list
from src.ingest.naive_chunker import run_section_chunking

logger = setup_logging("Stage4_Chunking")

def main():
    logger.info("Starting Stage 4: Section Chunking...")

    # 1. Check Stage 3 output exists
    nlp_chunk_files = get_file_list(DIR_NLP_CHUNKS, "json")
    logger.info(f"Found {len(nlp_chunk_files)} NLP chunk files from Stage 3.")

    if not nlp_chunk_files:
        logger.warning(
            f"No NLP chunk files found in {DIR_NLP_CHUNKS}. Run Stage 3 first."
        )
        return

    try:
        # 2. Run section chunking
        logger.info("Running section-based chunking...")
        stats = run_section_chunking()

        # 3. Verify output folder
        section_dir = DIR_FINAL_CHUNKS / "section"
        section_files = list(section_dir.glob("*.json")) if section_dir.exists() else []

        logger.info("Stage 4 Complete.")
        logger.info(f"Section chunks: {len(section_files)} files")
        logger.info(f"Output saved to: {section_dir}")

        # 4. Show per-book summary
        logger.info("Chunk statistics per book:")
        for book_id, count in stats.get("section_counts", {}).items():
            logger.info(f"  → {book_id}: {count} section chunks")

    except Exception as e:
        logger.error(f"Error in Stage 4 processing: {e}")
        raise

if __name__ == "__main__":
    main()
