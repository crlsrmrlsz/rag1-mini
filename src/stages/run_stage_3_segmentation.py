"""Stage 3: NLP segmentation of cleaned markdown files."""

import json
from pathlib import Path

from src.config import DIR_DEBUG_CLEAN, DIR_NLP_CHUNKS
from src.shared import setup_logging, get_file_list, get_output_path
from src.content_preparation.segmentation.nlp_segmenter import segment_document

logger = setup_logging("Stage3_Segmentation")


def main():
    """Run NLP segmentation pipeline."""
    logger.info("Starting Stage 3: NLP Segmentation")

    # Find cleaned markdown files
    input_files = get_file_list(DIR_DEBUG_CLEAN, "md")
    logger.info(f"Found {len(input_files)} cleaned Markdown files.")

    if not input_files:
        logger.warning(f"No files found in {DIR_DEBUG_CLEAN}. Run Stage 2 first.")
        return

    for md_path in input_files:
        logger.info(f"Processing: {md_path.name}")

        # Read and segment
        cleaned_text = md_path.read_text(encoding="utf-8")
        book_name = md_path.stem.replace("_debug", "")

        chunks = segment_document(cleaned_text, book_name)

        # Save JSON output
        json_path = get_output_path(md_path, DIR_DEBUG_CLEAN, DIR_NLP_CHUNKS, ".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        # Save Markdown output (human readable)
        md_out_path = get_output_path(md_path, DIR_DEBUG_CLEAN, DIR_NLP_CHUNKS)

        md_lines = [f"# Analyzed Content: {book_name}\n"]
        for i, chunk in enumerate(chunks):
            md_lines.append("---")
            md_lines.append(f"### Chunk {i+1}")
            md_lines.append(f"**Context:** `{chunk['context']}`")
            md_lines.append(f"**Sentences:** {chunk['num_sentences']}")
            for sent in chunk['sentences']:
                md_lines.append(f"- {sent}")
            md_lines.append("\n")

        md_out_path.write_text("\n".join(md_lines), encoding="utf-8")

        logger.info(f"Finished {md_path.name} -> {len(chunks)} chunks generated.")

    logger.info("Stage 3 complete.")


if __name__ == "__main__":
    main()
