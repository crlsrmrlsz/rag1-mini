"""PDF -> Markdown extractor with fallback to page-by-page.

Workflow:
- Attempt whole-document extraction using pymupdf4llm.to_markdown().
- If it fails, fallback to page-by-page extraction.
- On fallback, write PARTIAL_<name>.md including:
      - "## FAILED PAGES: X, Y, Z" header.
      - "## PAGE N MISSED" markers.
- Prints failed page summary to console.
"""

import logging
from pathlib import Path
from typing import List

import pymupdf.layout  # required before pymupdf4llm
import pymupdf4llm
import pymupdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PDFExtractor:
    """Simple CLI-only PDF → Markdown extractor with fallback strategy."""

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        header: bool = False,
        footer: bool = False,
        use_ocr: bool = False,
        force_text: bool = False,
    ):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.header = header
        self.footer = footer
        self.use_ocr = use_ocr
        self.force_text = force_text

    # ---- utility paths ----
    def get_output_path(self, pdf_path: Path) -> Path:
        rel = pdf_path.relative_to(self.raw_dir)
        return (self.processed_dir / rel).with_suffix(".md")

    def get_partial_output_path(self, pdf_path: Path) -> Path:
        rel = pdf_path.relative_to(self.raw_dir)
        folder = self.processed_dir / rel.parent
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"PARTIAL_{rel.stem}.md"

    def save(self, md: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")

    # ---- page-by-page fallback ----
    def extract_page_by_page(self, pdf_path: Path) -> None:
        failed: List[int] = []
        chunks: List[str] = []

        doc = pymupdf.open(str(pdf_path))
        total = doc.page_count

        for idx in range(total):
            try:
                logger.info("  Page %d/%d (fallback)", idx + 1, total)
                page_md = pymupdf4llm.to_markdown(
                    str(pdf_path),
                    pages=[idx],
                    page_chunks=True,
                    header=self.header,
                    footer=self.footer,
                    use_ocr=self.use_ocr,
                    force_text=self.force_text,
                )
                chunks.append(page_md)
            except Exception as e:
                logger.error("  FAILED page %d: %s", idx + 1, e)
                failed.append(idx + 1)
                chunks.append(f"\n\n## PAGE {idx + 1} MISSED\n\n")

        doc.close()

        # Build partial output
        failed_header = ""
        if failed:
            failed_header = "## FAILED PAGES: " + ", ".join(map(str, failed)) + "\n\n"

        md_text = failed_header + "\n".join(chunks)
        out_path = self.get_partial_output_path(pdf_path)
        self.save(md_text, out_path)

        # Console summary
        if failed:
            print(f"In {pdf_path.name}, the following pages failed: {failed}")
        else:
            print(f"In {pdf_path.name}, no pages failed.")

    # ---- single PDF ----
    def extract_single_pdf(self, pdf_path: Path) -> None:
        logger.info("Processing: %s", pdf_path)

        try:
            # whole-document extraction first
            md_text = pymupdf4llm.to_markdown(
                str(pdf_path),
                header=self.header,
                footer=self.footer,
                use_ocr=self.use_ocr,
                force_text=self.force_text,
            )

            out_path = self.get_output_path(pdf_path)
            self.save(md_text, out_path)
            logger.info("Saved: %s", out_path.relative_to(self.processed_dir))

        except Exception:
            logger.exception("Whole-document extraction failed; using page-by-page fallback.")
            self.extract_page_by_page(pdf_path)

    # ---- batch execution ----
    def find_pdfs(self) -> List[Path]:
        return sorted(self.raw_dir.rglob("*.pdf"))

    def extract_all(self) -> None:
        pdfs = self.find_pdfs()
        if not pdfs:
            logger.warning("No PDFs found in %s", self.raw_dir)
            return

        logger.info("Found %d PDF(s) to process.", len(pdfs))

        for p in pdfs:
            self.extract_single_pdf(p)

        logger.info("Extraction finished.")

def main() -> None:
    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    if not raw_dir.exists():
        logger.error("Raw directory not found: %s", raw_dir)
        return

    extractor = PDFExtractor(raw_dir, processed_dir, use_ocr=True, force_text=False)
    extractor.extract_all()


if __name__ == "__main__":
    main()
