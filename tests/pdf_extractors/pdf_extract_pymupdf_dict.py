import pymupdf  # PyMuPDF
import re
import os
from pathlib import Path

def clean_line(text):
    """Basic normalization: remove double spaces, stray hyphens, etc."""
    # De-hyphenate words split across lines ("motiva-\ntion" -> "motivation")
    text = re.sub(r"-\s+\n", "", text)
    text = text.replace("  ", " ")
    return text.strip()

def extract_page_simple(page, join_lines=True):
    """
    Extracts text from a page using PyMuPDF's 'dict' mode.
    Minimal interpretation: just blocks -> lines -> spans.
    Reading order is preserved as provided by the PDF.
    """
    data = page.get_text("dict")
    paragraphs = []
    current_para = []

    for block in data.get("blocks", []):
        if block["type"] != 0:   # type 0 = text
            continue

        for line in block.get("lines", []):
            line_text = "".join(span["text"] for span in line.get("spans", []))
            lt = clean_line(line_text)

            # Ignore empty lines
            if not lt:
                # finalize current paragraph if exists
                if current_para:
                    paragraphs.append(" ".join(current_para).strip())
                    current_para = []
                continue

            # Join lines into paragraphs (simple rule)
            if join_lines:
                current_para.append(lt)
            else:
                paragraphs.append(lt)

    # Flush final paragraph
    if current_para:
        paragraphs.append(" ".join(current_para).strip())

    return paragraphs


def create_debug_pdf(page, blocks, output_path):
    """Create a debug PDF with visual markers around text blocks, labeled by extraction order."""
    debug_doc = pymupdf.open()
    debug_page = debug_doc.new_page(width=page.rect.width, height=page.rect.height)

    debug_page.show_pdf_page(page.rect, page.parent, page.number)

    color = "red"  # Use red for text blocks

    for block_info in blocks:
        rect = pymupdf.Rect(*block_info["bbox"])
        debug_page.draw_rect(rect, color=pymupdf.utils.getColor(color), width=1)

        # Add order label
        text_point = pymupdf.Point(rect.x0 + 2, rect.y0 - 5)
        debug_page.insert_text(text_point, str(block_info["order"]), fontsize=8, color=pymupdf.utils.getColor(color))

    debug_doc.save(output_path)
    debug_doc.close()


def extract_document_simple_with_debug(filepath, debug_output=None):
    """
    Extract text from document with optional debug PDF output showing block boundaries.
    """
    doc = pymupdf.open(filepath)
    all_paragraphs = []

    for page_num, page in enumerate(doc, start=1):
        paras = extract_page_simple(page)
        for p in paras:
            all_paragraphs.append({"page": page_num, "text": p})

        if debug_output and page_num <= 5:
            data = page.get_text("dict")
            debug_blocks = []
            for order, block in enumerate(data.get("blocks", []), start=1):
                if block["type"] == 0:
                    debug_blocks.append({"bbox": block["bbox"], "order": order})
            debug_path = f"{debug_output}_page_{page_num}.pdf"
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            create_debug_pdf(page, debug_blocks, debug_path)
            print(f"Debug PDF saved: {debug_path}")

    doc.close()
    return all_paragraphs


# Original function for reference (kept but not used in main)
def extract_document_simple(filepath):
    doc = pymupdf.open(filepath)
    all_paragraphs = []

    for page_num, page in enumerate(doc, start=1):
        paras = extract_page_simple(page)
        for p in paras:
            all_paragraphs.append({"page": page_num, "text": p})

    doc.close()
    return all_paragraphs


# Example usage
if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Build paths relative to script location (common Python practice)
    pdf_file = script_dir.parent / "data" / "raw" / "ch1_ch14_Brain_and_behavior.pdf"
    debug_output_dir = script_dir.parent / "data" / "debug" / "pdf_extract_pymupdf_dict"
    debug_output_file = debug_output_dir / "pdf_extract_pymupdf_dict"
    
    # Convert to string for pymupdf (though Path objects usually work too)
    result = extract_document_simple_with_debug(
        str(pdf_file), 
        debug_output=str(debug_output_file)
    )

    print(f"Extracted {len(result)} paragraphs")
    for item in result[:20]:
        print(f"[Page {item['page']}] {item['text']}\n")
