import json
from pathlib import Path

from .config import DIR_MANUAL_REVIEW, DIR_DEBUG_CLEAN, DIR_FINAL_CHUNKS
from .utils import setup_logging, get_file_list, get_output_path
from .processors import run_structural_cleaning
from .processors import SemanticSegmenter

logger = setup_logging("Stage2_Processing")

def main():
    # 1. Initialize Segmenter
    logger.info("Initializing NLP Segmenter (SciSpaCy)...")
    segmenter = SemanticSegmenter()

    # 2. Find Reviewed Files
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
            cleaned_text = run_structural_cleaning(raw_text)

            # Save Debug Copy (STEP1_manual equivalent)
            debug_path = get_output_path(md_path, DIR_MANUAL_REVIEW, DIR_DEBUG_CLEAN, "_debug.md")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(cleaned_text, encoding="utf-8")

            # --- PHASE 4 & 5: NLP Segmentation & Filtering ---
            book_name = md_path.stem
            chunks = segmenter.process_document(cleaned_text, book_name)

            # Save Final JSON (Machine Readable)
            json_path = get_output_path(md_path, DIR_MANUAL_REVIEW, DIR_FINAL_CHUNKS, ".json")
            json_path.parent.mkdir(parents=True, exist_ok=True)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)

            # Save Final Markdown (Human Readable / RAG Input)
            md_out_path = get_output_path(md_path, DIR_MANUAL_REVIEW, DIR_FINAL_CHUNKS, "_final.md")

            md_lines = [f"# Analyzed Content: {md_path.stem}\n"]
            for i, chunk in enumerate(chunks):
                md_lines.append("---")
                md_lines.append(f"### Chunk {i+1}")
                md_lines.append(f"**Context:** `{chunk['context']}`")
                md_lines.append(f"**Sentences:** {chunk['num_sentences']}")
                for sent in chunk['sentences']:
                    md_lines.append(f"- {sent}")
                md_lines.append("\n")

            md_out_path.write_text("\n".join(md_lines), encoding="utf-8")

            logger.info(f"✔ Finished {md_path.name} -> {len(chunks)} chunks generated.")

        except Exception as e:
            logger.error(f"✘ Error processing {md_path.name}: {e}")

    logger.info("Stage 2 Complete.")

if __name__ == "__main__":
    main()
