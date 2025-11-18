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
    """Remove top/bottom repeated page elements."""
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
    raw_blocks = page.get_text("blocks")

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

        for b, col in zip(filtered, labels):
            b["col"] = int(col)

    # --- Step 4: sort inside each column ---------------------
    paragraphs = []
    for col_id in sorted({b["col"] for b in filtered}):
        col_blocks = [b for b in filtered if b["col"] == col_id]
        col_blocks.sort(key=lambda b: (b["y0"], b["x0"]))
        merged = merge_blocks(col_blocks)
        paragraphs.extend(merged)

    # --- Step 5: final global sort (top → bottom) ------------
    paragraphs.sort(key=lambda b: (b["y0"], b["x0"]))

    return [p["text"] for p in paragraphs]


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
    results = extract_document_text(filepath)

    for r in results[:30]:
        print(f"[Page {r['page']}] {r['text']}\n")
