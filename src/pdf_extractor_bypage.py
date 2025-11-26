import logging
from pathlib import Path
from typing import List, Dict
import json

import pymupdf.layout
from pymupdf4llm import parse_document

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFExtractor:

    def __init__(self, raw_dir: Path, processed_dir: Path):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.failed_pages: Dict[str, List[int]] = {}

    def find_pdfs(self) -> List[Path]:
        return sorted(self.raw_dir.rglob("*.pdf"))

    def get_output_base(self, pdf_path: Path) -> Path:
        relative = pdf_path.relative_to(self.raw_dir)
        return (self.processed_dir / relative).with_suffix("")

    def extract_single_pdf(self, pdf_path: Path) -> bool:
        """Extract PDF page by page with robust error handling."""
        logger.info(f"Processing {pdf_path}")

        # First, get total page count
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            total_pages = doc.page_count
            doc.close()
        except Exception as e:
            logger.error(f"Failed to open {pdf_path}: {e}")
            return False

        base = self.get_output_base(pdf_path)
        base.mkdir(parents=True, exist_ok=True)

        successful_pages = []
        failed_pages = []

        # Process each page individually
        for page_num in range(total_pages):
            try:
                # Parse just this one page
                pdoc = parse_document(
                    pdf_path,
                    pages=page_num,  # Single page
                    use_ocr=False,
                    embed_images=False,
                    write_images=False,
                    show_progress=False
                )

                # Extract text from the single page
                page_text = pdoc.to_text(header=False, footer=False)
                
                # Save page text
                page_path = base.parent / f"{base.name}_page_{page_num + 1:04d}.txt"
                page_path.write_text(page_text, encoding="utf-8")
                
                successful_pages.append(page_num + 1)

            except Exception as e:
                logger.error(f"Failed at page {page_num + 1} of {pdf_path}: {e}")
                failed_pages.append(page_num + 1)
                continue

        # Save metadata about the extraction
        metadata = {
            "filename": pdf_path.name,
            "total_pages": total_pages,
            "successful_pages": len(successful_pages),
            "failed_pages": failed_pages,
        }
        
        metadata_path = base.with_suffix(".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if failed_pages:
            self.failed_pages[str(pdf_path)] = failed_pages
            logger.warning(
                f"Completed {pdf_path.name}: "
                f"{len(successful_pages)}/{total_pages} pages ok, "
                f"{len(failed_pages)} pages failed: {failed_pages}"
            )
        else:
            logger.info(
                f"Completed {pdf_path.name}: "
                f"all {total_pages} pages processed successfully"
            )

        return len(successful_pages) > 0

    def extract_all(self):
        """Extract all PDFs and generate a failure report."""
        pdfs = self.find_pdfs()
        if not pdfs:
            logger.warning("No PDFs found.")
            return

        ok = 0
        fail = 0

        for pdf in pdfs:
            if self.extract_single_pdf(pdf):
                ok += 1
            else:
                fail += 1

        # Save failure report
        if self.failed_pages:
            report_path = self.processed_dir / "failed_pages_report.json"
            report_path.write_text(
                json.dumps(self.failed_pages, indent=2),
                encoding="utf-8"
            )
            logger.info(f"Failure report saved to {report_path}")

        logger.info(
            f"Extraction finished: "
            f"{ok} documents ok, {fail} documents completely failed."
        )


def main():
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    extractor = PDFExtractor(raw_dir, processed_dir)
    extractor.extract_all()


if __name__ == "__main__":
    main()