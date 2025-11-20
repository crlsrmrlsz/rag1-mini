import pymupdf  # PyMuPDF
import re
import os
from pathlib import Path
from collections import defaultdict

def clean_line(text):
    """Basic normalization: remove double spaces, stray hyphens, etc."""
    # De-hyphenate words split across lines ("motiva-\ntion" -> "motivation")
    text = re.sub(r"-\s+\n", "", text)
    text = text.replace("  ", " ")
    return text.strip()


def detect_header_footer_margins(doc, sample_pages=5):
    """
    Auto-detect header/footer margins by finding repeated content
    across multiple pages at consistent y-positions.
    Returns: (top_margin, bottom_margin) in points
    """
    top_texts = defaultdict(list)  # y_position -> list of (page, text)
    bottom_texts = defaultdict(list)
    
    sample_pages = min(sample_pages, len(doc))
    
    for page_num in range(sample_pages):
        page = doc[page_num]
        page_height = page.rect.height
        blocks = page.get_text("blocks")
        
        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            text = text.strip()
            if not text:
                continue
            
            # Check if in top 10% of page
            if y0 < page_height * 0.1:
                top_texts[round(y0, 1)].append((page_num, text))
            
            # Check if in bottom 10% of page
            if y1 > page_height * 0.9:
                bottom_texts[round(y1, 1)].append((page_num, text))
    
    # Find repeated content (appears on multiple pages)
    top_margin = 0
    bottom_margin = 0
    
    for y_pos, occurrences in top_texts.items():
        if len(occurrences) >= 2:  # Appears on at least 2 pages
            top_margin = max(top_margin, y_pos + 20)  # Add buffer
    
    for y_pos, occurrences in bottom_texts.items():
        if len(occurrences) >= 2:
            page_height = doc[0].rect.height
            bottom_margin = max(bottom_margin, page_height - y_pos + 20)
    
    # Use sensible defaults if auto-detection fails
    if top_margin == 0:
        top_margin = 60  # Default: 60pt from top
    if bottom_margin == 0:
        bottom_margin = 60  # Default: 60pt from bottom
    
    return top_margin, bottom_margin


def detect_columns(blocks, page_width, min_column_width=100):
    """
    Auto-detect column structure by analyzing x-coordinates of text blocks.
    Returns: list of column boundaries [(x_start, x_end), ...]
    """
    if not blocks:
        return [(0, page_width)]
    
    # Collect all x-coordinates
    x_starts = sorted([b["x0"] for b in blocks])
    x_ends = sorted([b["x1"] for b in blocks])
    
    # Find gaps in x-coordinates (potential column boundaries)
    gaps = []
    for i in range(len(x_ends) - 1):
        gap_start = x_ends[i]
        gap_end = x_starts[i + 1]
        gap_width = gap_end - gap_start
        
        # If there's a significant gap, it's likely a column boundary
        if gap_width > 20:  # At least 20pt gap
            gaps.append((gap_start, gap_end))
    
    # If no clear gaps found, assume single column
    if not gaps:
        return [(0, page_width)]
    
    # Build column boundaries
    columns = []
    current_x = 0
    
    for gap_start, gap_end in gaps:
        # Check if column is wide enough
        col_width = gap_start - current_x
        if col_width >= min_column_width:
            columns.append((current_x, gap_start))
            current_x = gap_end
    
    # Add final column
    if page_width - current_x >= min_column_width:
        columns.append((current_x, page_width))
    
    # If analysis failed, return single color
    if not columns:
        return [(0, page_width)]
    
    return columns


def assign_blocks_to_columns(blocks, columns):
    """
    Assign each text block to its appropriate column based on x-position.
    Returns: dict of {column_index: [blocks]}
    """
    column_blocks = {i: [] for i in range(len(columns))}
    
    for block in blocks:
        block_center_x = (block["x0"] + block["x1"]) / 2
        
        # Find which column this block belongs to
        for col_idx, (col_start, col_end) in enumerate(columns):
            if col_start <= block_center_x <= col_end:
                column_blocks[col_idx].append(block)
                break
    
    return column_blocks


def extract_page_multicolumn(page, top_margin, bottom_margin):
    """
    Extract text from a page using multi-column detection.
    Auto-detects columns and processes in proper reading order.
    """
    page_height = page.rect.height
    page_width = page.rect.width
    
    # Get all text blocks
    raw_blocks = page.get_text("blocks", sort=False)
    blocks = []
    
    for b in raw_blocks:
        x0, y0, x1, y1, text, *_ = b
        text = text.strip()
        if not text:
            continue
        
        # Filter out header/footer based on detected margins
        if y0 < top_margin or y1 > (page_height - bottom_margin):
            continue
        
        blocks.append({
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "text": text
        })
    
    if not blocks:
        return []
    
    # Auto-detect columns
    columns = detect_columns(blocks, page_width)
    
    # Assign blocks to columns
    column_blocks = assign_blocks_to_columns(blocks, columns)
    
    # Process each column in order, top to bottom
    paragraphs = []
    for col_idx in sorted(column_blocks.keys()):
        col_blocks = column_blocks[col_idx]
        
        # Sort blocks in this column by vertical position
        col_blocks.sort(key=lambda b: (b["y0"], b["x0"]))
        
        # Extract text from blocks
        for block in col_blocks:
            text = clean_line(block["text"])
            if text:
                paragraphs.append(text)
    
    return paragraphs, columns, blocks


def create_debug_pdf(page, blocks, columns, output_path):
    """
    Create a debug PDF showing detected columns and block boundaries.
    """
    debug_doc = pymupdf.open()
    debug_page = debug_doc.new_page(width=page.rect.width, height=page.rect.height)
    
    # Draw original page
    debug_page.show_pdf_page(page.rect, page.parent, page.number)
    
    # Draw column boundaries in green
    for col_idx, (col_start, col_end) in enumerate(columns):
        rect = pymupdf.Rect(col_start, 0, col_end, page.rect.height)
        debug_page.draw_rect(rect, color=pymupdf.utils.getColor("green"), width=2, dashes="[3] 0")
        
        # Add column label
        text_point = pymupdf.Point(col_start + 5, 20)
        debug_page.insert_text(text_point, f"Col {col_idx + 1}", fontsize=10, 
                              color=pymupdf.utils.getColor("green"))
    
    # Draw block boundaries in green
    for order, block in enumerate(blocks, start=1):
        rect = pymupdf.Rect(block["x0"], block["y0"], block["x1"], block["y1"])
        debug_page.draw_rect(rect, color=pymupdf.utils.getColor("green"), width=1)
        
        # Add order label
        text_point = pymupdf.Point(block["x0"] + 2, block["y0"] - 5)
        debug_page.insert_text(text_point, str(order), fontsize=8, 
                              color=pymupdf.utils.getColor("green"))
    
    debug_doc.save(output_path)
    debug_doc.close()


def extract_document_multicolumn_with_debug(filepath, debug_output=None):
    """
    Extract text from document using multi-column detection.
    Auto-detects columns, header/footer margins, and processes in reading order.
    """
    doc = pymupdf.open(filepath)
    
    # Auto-detect header/footer margins
    print("Auto-detecting header/footer margins...")
    top_margin, bottom_margin = detect_header_footer_margins(doc)
    print(f"Detected margins: top={top_margin:.1f}pt, bottom={bottom_margin:.1f}pt")
    
    all_paragraphs = []
    
    for page_num, page in enumerate(doc, start=1):
        paras, columns, blocks = extract_page_multicolumn(page, top_margin, bottom_margin)
        
        for p in paras:
            all_paragraphs.append({"page": page_num, "text": p})
        
        # Create debug visualization for first 5 pages
        if debug_output and page_num <= 5:
            debug_path = f"{debug_output}_page_{page_num}.pdf"
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            create_debug_pdf(page, blocks, columns, debug_path)
            print(f"Debug PDF saved: {debug_path} (detected {len(columns)} columns)")
    
    doc.close()
    return all_paragraphs


# Example usage
if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Build paths relative to project root
    pdf_file = script_dir.parent.parent / "data" / "raw" / "ch1_ch14_Brain_and_behavior.pdf"
    debug_output_dir = script_dir.parent.parent / "data" / "debug" / "pdf_extract_pymupdf_multicolumn"
    debug_output_file = debug_output_dir / "pdf_extract_pymupdf_multicolumn"
    
    print(f"Processing PDF: {pdf_file}")
    result = extract_document_multicolumn_with_debug(
        str(pdf_file), 
        debug_output=str(debug_output_file)
    )
    
    print(f"\nExtracted {len(result)} paragraphs")
    print("\nFirst 10 paragraphs:")
    for item in result[:10]:
        print(f"[Page {item['page']}] {item['text'][:100]}...")
