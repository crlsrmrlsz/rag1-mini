from pathlib import Path

from .config import DIR_MANUAL_REVIEW, DIR_DEBUG_CLEAN
from .utils import setup_logging, get_file_list, get_output_path
from .processors import run_structural_cleaning, setup_cleaning_logger


logger = setup_logging("Stage2_Processing")

def main():
    logger.info("Setting up cleaning logger...")
    cleaning_logger = setup_cleaning_logger()

    # 1. Find Reviewed Files
    input_files = get_file_list(DIR_MANUAL_REVIEW, "md")
    logger.info(f"Found {len(input_files)} reviewed Markdown files.")

    if not input_files:
        logger.warning(f"No files found in {DIR_MANUAL_REVIEW}. Did you move the files after review?")
        return

    for md_path in input_files:
        try:
            logger.info(f"Processing: {md_path.name}")

            # --- PHASE 3: Structural Cleaning ---
            raw_text = md_path.read_text(encoding="utf-8")
            book_name = md_path.stem
            cleaned_text, cleaning_log = run_structural_cleaning(
                raw_text,
                book_name=book_name,
                enable_logging=True
            )

            # Log the cleaning report
            if cleaning_log:
                cleaning_logger.info(cleaning_log.generate_report())
                logger.info(f"  → Cleaning changes: {len(cleaning_log.lines_removed)} lines removed, "
                            f"{len(cleaning_log.inline_removals)} inline removals, "
                            f"{cleaning_log.paragraphs_merged} paragraphs merged")

            # Save cleaned copy with same name and extension to DIR_DEBUG_CLEAN
            debug_path = get_output_path(md_path, DIR_MANUAL_REVIEW, DIR_DEBUG_CLEAN)  # No new_extension - preserves filename
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(cleaned_text, encoding="utf-8")

            logger.info(f"✔ Finished cleaning {md_path.name}")

        except Exception as e:
            logger.error(f"✘ Error processing {md_path.name}: {e}")

    logger.info("Stage 2 Complete (Markdown Cleaning).")

if __name__ == "__main__":
    main()
