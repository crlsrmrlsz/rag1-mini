import pymupdf
import re
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import os

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

CAPTION_REGEX = re.compile(
    r"^(fig\.?|figure)\s?\d+", 
    re.IGNORECASE
)

# Magic numbers extracted as constants for better maintainability
HEADER_FOOTER_MARGIN = 60
MAX_VERTICAL_GAP = 12
MIN_X_DIFFERENCE_MULTICOLUMN = 40
MAX_CLUSTERS = 3
KMEANS_INIT = 5
DEBUG_PAGES_LIMIT = 5

# Colors for different columns in debug visualization
COLUMN_COLORS = ["red", "green", "blue", "orange", "purple"]  # up to 5 columns

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def is_caption(text: str) -> bool:
    """Detect figure captions (common academic patterns)."""
    text = text.strip()
    if len(text) < 5:
        return False
    return bool(CAPTION_REGEX.search(text))

def is_header_footer(block, page_height, margin=HEADER_FOOTER_MARGIN):
    """Remove top/bottom repeated page elements.
       Important: In PyMuPDF (fitz), page dimensions are returned in points, not pixels. 
       PDF standard uses 72 points per inch.
       So for a standard A4 PDF:
       Width: 210 mm ≈ 8.27 inches → 8.27 × 72 ≈ 595 points
       Height: 297 mm ≈ 11.69 inches → 11.69 × 72 ≈ 842 points
       60 / 72 = 0.83 inches ≈ 21 mm from top or bottom."""
    x0, y0, x1, y1, *_ = block
    return (y0 < margin) or (y1 > (page_height - margin))

def merge_blocks(blocks, max_gap=MAX_VERTICAL_GAP):
    """Merge vertically adjacent blocks that appear to be the same paragraph."""
    if not blocks:
        return []

    merged = []
    cur = blocks[0]

    for nxt in blocks[1:]:
        same_left = abs(cur["x0"] - nxt["x0"]) < 15
        vertical_close = (nxt["y0"] - cur["y1"]) < max_gap

        if same_left and vertical_close:
            cur["text"] = cur["text"].rstrip() + " " + nxt["text"].lstrip()
            cur["y1"] = nxt["y1"]
        else:
            merged.append(cur)
            cur = nxt

    merged.append(cur)
    return merged

def detect_columns(filtered_blocks):
    """Detect column structure using KMeans clustering."""
    x_positions = np.array([[b["x0"]] for b in filtered_blocks])

    # If all x0 are within threshold, assume one column
    if max(x_positions) - min(x_positions) < MIN_X_DIFFERENCE_MULTICOLUMN:
        for b in filtered_blocks:
            b["col"] = 0
        return filtered_blocks

    # Try multiple clusters (2-3)
    n_clusters = min(MAX_CLUSTERS, len(filtered_blocks))
    kmeans = KMeans(n_clusters=n_clusters, n_init=KMEANS_INIT, random_state=0)
    labels = kmeans.fit_predict(x_positions)

    # Remap KMeans labels to actual column positions (left-to-right)
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_centers = np.argsort(cluster_centers)  # indices in left-to-right order

    # Create mapping from KMeans label to actual column number
    label_to_col = {}
    for actual_col, kmeans_label in enumerate(sorted_centers):
        label_to_col[kmeans_label] = actual_col

    for b, label in zip(filtered_blocks, labels):
        b["col"] = label_to_col[label]

    return filtered_blocks

def extract_raw_blocks(page):
    """Extract raw text blocks from page with proper sorting."""
    raw_blocks = page.get_text("blocks", sort=True)
    blocks = []

    for b in raw_blocks:
        x0, y0, x1, y1, text, *_ = b
        t = text.strip()
        if not t:
            continue

        blocks.append({
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "text": t
        })

    return blocks

def filter_blocks(blocks, page_height):
    """Remove headers, footers, and captions from blocks."""
    filtered = []
    for b in blocks:
        if is_header_footer((b["x0"], b["y0"], b["x1"], b["y1"]), page_height):
            continue
        if is_caption(b["text"]):
            continue
        filtered.append(b)

    return filtered

def extract_text_from_columns(filtered_blocks):
    """Extract text from columns in proper reading order."""
    if not filtered_blocks:
        return []

    # Detect column structure
    detect_columns(filtered_blocks)

    # Process each column
    paragraphs = []
    for col_id in sorted({b["col"] for b in filtered_blocks}):
        col_blocks = [b for b in filtered_blocks if b["col"] == col_id]
        col_blocks.sort(key=lambda b: (b["y0"], b["x0"]))
        merged = merge_blocks(col_blocks)
        paragraphs.extend(merged)

    return [p["text"] for p in paragraphs]

# ------------------------------------------------------------
# Main Functions
# ------------------------------------------------------------

def extract_clean_text_from_page(page):
    """Extract clean text from a single page."""
    page_height = page.rect.height

    # Step 1: Extract raw blocks
    raw_blocks = extract_raw_blocks(page)

    if not raw_blocks:
        return []

    # Step 2: Filter out headers, footers, captions
    filtered_blocks = filter_blocks(raw_blocks, page_height)

    if not filtered_blocks:
        return []

    # Step 3: Extract text from columns in proper reading order
    return extract_text_from_columns(filtered_blocks)

def create_debug_pdf(page, blocks, output_path):
    """Create a debug PDF with visual markers for blocks and columns."""
    # Create a new PDF with the same page
    debug_doc = pymupdf.open()
    debug_page = debug_doc.new_page(width=page.rect.width, height=page.rect.height)

    # Draw the original page content
    debug_page.show_pdf_page(page.rect, page.parent, page.number)

    # Draw rectangles around blocks with different colors for columns
    for block in blocks:
        col = block.get("col", 0)
        color = COLUMN_COLORS[col % len(COLUMN_COLORS)]

        # Draw rectangle
        rect = pymupdf.Rect(block["x0"], block["y0"], block["x1"], block["y1"])
        debug_page.draw_rect(rect, color=pymupdf.utils.getColor(color), width=1)

        # Add column label
        text_point = pymupdf.Point(block["x0"] + 2, block["y0"] - 5)
        debug_page.insert_text(text_point, f"Col {col}", fontsize=8, color=pymupdf.utils.getColor(color))

    debug_doc.save(output_path)
    debug_doc.close()

def extract_clean_text_with_debug(filepath, debug_output=None):
    """Extract text with optional debug visualization."""
    doc = pymupdf.open(filepath)
    output = []

    for page_num, page in enumerate(doc, start=1):
        # Get blocks for debug visualization
        raw_blocks = page.get_text("blocks", sort=True)
        debug_blocks = []

        page_height = page.rect.height

        for b in raw_blocks:
            x0, y0, x1, y1, text, *_ = b
            t = text.strip()
            if not t:
                continue
            if is_header_footer((x0, y0, x1, y1), page_height):
                continue
            if is_caption(t):
                continue

            debug_blocks.append({
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "text": t
            })

        # Assign column labels for debug visualization
        if debug_blocks:
            x_positions = np.array([[b["x0"]] for b in debug_blocks])

            if max(x_positions) - min(x_positions) >= MIN_X_DIFFERENCE_MULTICOLUMN:
                n_clusters = min(MAX_CLUSTERS, len(debug_blocks))
                kmeans = KMeans(n_clusters=n_clusters, n_init=KMEANS_INIT, random_state=0)
                labels = kmeans.fit_predict(x_positions)

                # Remap labels to actual column positions
                cluster_centers = kmeans.cluster_centers_.flatten()
                sorted_centers = np.argsort(cluster_centers)
                label_to_col = {}
                for actual_col, kmeans_label in enumerate(sorted_centers):
                    label_to_col[kmeans_label] = actual_col

                for b, label in zip(debug_blocks, labels):
                    b["col"] = label_to_col[label]
            else:
                for b in debug_blocks:
                    b["col"] = 0

        # Extract clean text
        paragraphs = extract_clean_text_from_page(page)
        for para in paragraphs:
            output.append({
                "page": page_num,
                "text": para
            })

        # Create debug PDF for first few pages if requested
        if debug_output and page_num <= DEBUG_PAGES_LIMIT:
            debug_path = f"{debug_output}_page_{page_num}.pdf"
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            create_debug_pdf(page, debug_blocks, debug_path)
            print(f"Debug PDF saved: {debug_path}")

    doc.close()
    return output

def extract_document_text(filepath):
    """Extract text from entire document."""
    doc = pymupdf.open(filepath)
    output = []

    for page_num, page in enumerate(doc, start=1):
        paragraphs = extract_clean_text_from_page(page)
        for para in paragraphs:
            output.append({
                "page": page_num,
                "text": para
            })

    doc.close()
    return output

# ------------------------------------------------------------
# Example
# ------------------------------------------------------------

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Build paths relative to script location (common Python practice)
    pdf_file = script_dir.parent / "data" / "raw" / "ch1_ch14_Brain_and_behavior.pdf"
    debug_output_dir = script_dir.parent / "data" / "debug" / "pdf_extract_pymupdf_blocks" / "pdf_extract_pymupdf_blocks"
    
    # Test extraction with debug visualization
    print("Testing text extraction with debug visualization...")
    results = extract_clean_text_with_debug(str(pdf_file), debug_output=str(debug_output_dir))

    print(f"\nExtracted {len(results)} text blocks")

    print("\nFirst 30 text blocks:")
    for r in results[:30]:
        print(f"[Page {r['page']}] {r['text']}\n")

    # Also test basic extraction for comparison
    print("\n=== BASIC EXTRACTION FOR COMPARISON ===")
    basic_results = extract_document_text(str(pdf_file))

    print("Basic extraction comparison (first 10 blocks):")
    for r in basic_results[:10]:
        print(f"[Page {r['page']}] {r['text'][:100]}...")
