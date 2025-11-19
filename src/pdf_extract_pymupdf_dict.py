import pymupdf  # PyMuPDF
import re

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
    filepath = "../data/raw/ch1_ch14_Brain_and_behavior.pdf"
    result = extract_document_simple(filepath)

    print(f"Extracted {len(result)} paragraphs")
    for item in result[:20]:
        print(f"[Page {item['page']}] {item['text']}\n")
