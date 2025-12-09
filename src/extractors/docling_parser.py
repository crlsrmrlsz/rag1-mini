from docling.datamodel.document import InputFormat, DocItemLabel
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from src.utils import setup_logging

logger = setup_logging(__name__)

class DoclingExtractor:
    def __init__(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False 

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def _get_all_descendants(self, item):
        """Recursively collect all children, grandchildren, etc."""
        descendants = []
        # Check if the item has children
        if hasattr(item, "children") and item.children:
            for child in item.children:
                descendants.append(child)
                # Recursively get children of the child
                descendants.extend(self._get_all_descendants(child))
        return descendants

    def convert_pdf(self, pdf_path) -> str:
        try:
            result = self.converter.convert(pdf_path)
            doc = result.document

            # 1. First removal step: Remove captions, footnotes, page headers/footers, and tables
            items_to_remove = []
            labels_to_remove = {DocItemLabel.CAPTION, DocItemLabel.FOOTNOTE, 
                                DocItemLabel.PAGE_FOOTER, DocItemLabel.PAGE_HEADER, 
                                DocItemLabel.TABLE}

            for item, level in doc.iterate_items():
                # Check if the item has a label and if it matches our target list
                if hasattr(item, "label") and item.label in labels_to_remove:
                    items_to_remove.append(item)

            # Delete the items from the document
            # This updates the document tree in-place
            if items_to_remove:
                doc.delete_items(node_items=items_to_remove)

            # 2. Second removal step: Remove pictures and all their children
            items_to_remove = []   # Use a list instead of a set
            seen_ids = set()       # Track IDs to avoid duplicates

            for item, level in doc.iterate_items():
                # Check if it is a Picture
                if hasattr(item, "label") and item.label == DocItemLabel.PICTURE:
                    
                    # A. Add the Picture item itself (if not already added)
                    if id(item) not in seen_ids:
                        items_to_remove.append(item)
                        seen_ids.add(id(item))
                    
                    # B. Get all children (captions, texts inside)
                    children = self._get_all_descendants(item)
                    
                    for child in children:
                        if id(child) not in seen_ids:
                            items_to_remove.append(child)
                            seen_ids.add(id(child))

            # Delete the items
            if items_to_remove:
                logger.info(f"Removing {len(items_to_remove)} items...")
                doc.delete_items(node_items=items_to_remove)

            return doc.export_to_markdown()

        except Exception as e:
            logger.error(f"Failed to convert {pdf_path}: {e}")
            raise e
