import pymupdf
import re
import numpy as np
from sklearn.cluster import KMeans

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

CAPTION_REGEX = re.compile(
    r"^(fig\.?|figure)\s?\d+", 
    re.IGNORECASE
)

def is_caption(text: str) -> bool:
    """Detect figure captions (common academic patterns)."""
    text = text.strip()
    if len(text) < 5:
        return False
    return bool(CAPTION_REGEX.search(text))

def is_header_footer(block, page_height, margin=60):
    """Remove top/bottom repeated page elements.
       Important: In PyMuPDF (fitz), page dimensions are returned in points, not pixels. 
       PDF standard uses 72 points per inch.
       So for a standard A4 PDF:
       Width: 210 mm ≈ 8.27 inches → 8.27 × 72 ≈ 595 points
       Height: 297 mm ≈ 11.69 inches → 11.69 × 72 ≈ 842 points
       60 / 72 = 0.83 inches ≈ 21 mm from top or bottom."""
    x0, y0, x1, y1, *_ = block
    return (y0 < margin) or (y1 > (page_height - margin))

def merge_blocks(blocks, max_gap=12):
    """
    Merge vertically adjacent blocks that appear to be the same paragraph.
    Slightly more forgiving than the previous version.
    """

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


# ------------------------------------------------------------
# Page-level extraction
# ------------------------------------------------------------

def extract_clean_text_from_page(page):
    page_height = page.rect.height

    # --- Step 1: raw blocks ---------------------------------
    # Use PyMuPDF's built-in sorting for proper reading order (top-to-bottom, left-to-right)
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

    if not blocks:
        return []

    # --- Step 2: remove headers, footers, captions ----------
    filtered = []
    for b in blocks:
        if is_header_footer((b["x0"], b["y0"], b["x1"], b["y1"]), page_height):
            continue
        if is_caption(b["text"]):
            continue
        filtered.append(b)

    if not filtered:
        return []

    # --- Step 3: detect column structure ---------------------
    x_positions = np.array([[b["x0"]] for b in filtered])

    # if all x0 are within 40px, assume one column
    if max(x_positions) - min(x_positions) < 40:
        for b in filtered:
            b["col"] = 0
    else:
        # try 2–3 clusters
        n_clusters = min(3, len(filtered))
        kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
        labels = kmeans.fit_predict(x_positions)

        # Remap KMeans labels to actual column positions (left-to-right)
        cluster_centers = kmeans.cluster_centers_.flatten()
        sorted_centers = np.argsort(cluster_centers)  # indices in left-to-right order

        # Create mapping from KMeans label to actual column number
        label_to_col = {}
        for actual_col, kmeans_label in enumerate(sorted_centers):
            label_to_col[kmeans_label] = actual_col

        for b, label in zip(filtered, labels):
            b["col"] = label_to_col[label]

    # --- Step 4: sort inside each column ---------------------
    paragraphs = []
    for col_id in sorted({b["col"] for b in filtered}):
        col_blocks = [b for b in filtered if b["col"] == col_id]
        col_blocks.sort(key=lambda b: (b["y0"], b["x0"]))
        merged = merge_blocks(col_blocks)
        paragraphs.extend(merged)

    # --- Step 5: paragraphs are already in correct reading order (left column to right, top to bottom within column)

    return [p["text"] for p in paragraphs]


def create_debug_pdf(page, blocks, output_path):
    """Create a debug PDF with visual markers for blocks and columns."""
    import pymupdf as fitz

    # Create a new PDF with the same page
    debug_doc = fitz.open()
    debug_page = debug_doc.new_page(width=page.rect.width, height=page.rect.height)

    # Draw the original page content
    debug_page.show_pdf_page(page.rect, page.parent, page.number)

    # Draw rectangles around blocks with different colors for columns
    colors = ["red", "green", "blue", "orange", "purple"]  # up to 5 columns

    for block in blocks:
        col = block.get("col", 0)
        color = colors[col % len(colors)]

        # Draw rectangle
        rect = fitz.Rect(block["x0"], block["y0"], block["x1"], block["y1"])
        debug_page.draw_rect(rect, color=fitz.utils.getColor(color), width=1)

        # Add column label
        text_point = fitz.Point(block["x0"] + 2, block["y0"] - 5)
        debug_page.insert_text(text_point, f"Col {col}", fontsize=8, color=fitz.utils.getColor(color))

    debug_doc.save(output_path)
    debug_doc.close()


def extract_clean_text_with_debug(filepath, debug_output=None):
    """Extract text with optional debug visualization."""
    doc = pymupdf.open(filepath)
    output = []

    for page_num, page in enumerate(doc, start=1):
        # Get blocks for debug visualization
        # sort=True ensures proper reading order, vertical, then horizontal
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

            if max(x_positions) - min(x_positions) >= 40:
                n_clusters = min(3, len(debug_blocks))
                kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
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
        if debug_output and page_num <= 5:  # Only first 5 pages for debugging
            debug_path = f"{debug_output}_page_{page_num}.pdf"
            import os
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            create_debug_pdf(page, debug_blocks, debug_path)
            print(f"Debug PDF saved: {debug_path}")

    doc.close()
    return output


# ------------------------------------------------------------
# Full-document extraction
# ------------------------------------------------------------

def extract_document_text(filepath):
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
    filepath = "../data/raw/ch1_ch14_Brain_and_behavior.pdf"

    # Test extraction with debug visualization
    print("Testing text extraction with debug visualization...")
    results = extract_clean_text_with_debug(filepath, debug_output="../data/debug/debug_visualization")

    print(f"\nExtracted {len(results)} text blocks")

    print("\nFirst 30 text blocks:")
    for r in results[:30]:
        print(f"[Page {r['page']}] {r['text']}\n")

    # Also test basic extraction for comparison
    print("\n=== BASIC EXTRACTION FOR COMPARISON ===")
    basic_results = extract_document_text(filepath)

    print("Basic extraction comparison (first 10 blocks):")
    for r in basic_results[:10]:
        print(f"[Page {r['page']}] {r['text'][:100]}...")
