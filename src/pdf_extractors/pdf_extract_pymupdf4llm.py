import pymupdf  # PyMuPDF
import pymupdf4llm  # PyMuPDF4LLM
import re
import os
from pathlib import Path
import json

def clean_line(text):
    """Basic normalization: remove double spaces, stray hyphens, etc."""
    # De-hyphenate words split across lines ("motiva-\ntion" -> "motivation")
    text = re.sub(r"-\s+\n", "", text)
    text = text.replace("  ", " ")
    return text.strip()


def extract_text_from_markdown(markdown_content, page_mappings):
    """
    Convert markdown content to our standard paragraph format.
    PyMuPDF4LLM often preserves structure, so we need to break it into paragraphs.
    """
    paragraphs = []
    
    # Split by double newlines to get paragraphs/sections
    sections = re.split(r'\n\s*\n', markdown_content)
    
    for section in sections:
        if not section.strip():
            continue
            
        # Clean the section
        section = section.strip()
        
        # Handle headers - if it's a header, treat it as a separate paragraph
        if section.startswith('#'):
            paragraphs.append(section)
        else:
            # Split long paragraphs into smaller ones if they contain multiple sentences
            sentences = re.split(r'[.!?]+', section)
            current_para = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # If adding this sentence would make the paragraph too long, start a new one
                if current_para and len(current_para + sentence) > 200:
                    if current_para.strip():
                        paragraphs.append(current_para.strip())
                    current_para = sentence
                else:
                    if current_para:
                        current_para += ". " + sentence
                    else:
                        current_para = sentence
            
            # Add the last paragraph
            if current_para.strip():
                paragraphs.append(current_para.strip())
    
    # Add page numbers based on mapping
    result = []
    for i, para in enumerate(paragraphs):
        page_num = page_mappings.get(i, 1)  # Default to page 1 if not mapped
        result.append({"page": page_num, "text": para})
    
    return result


def create_debug_pdf_llm(doc, page_num, markdown_content, output_path):
    """
    Create a debug PDF showing a specific page with extraction visualization.
    """
    if not doc or page_num > len(doc):
        return
        
    page = doc[page_num - 1]  # Convert to 0-based index
        
    debug_doc = pymupdf.open()
    debug_page = debug_doc.new_page(width=page.rect.width, height=page.rect.height)
    
    # Draw the specific page
    debug_page.show_pdf_page(page.rect, page.parent, page.number)
    
    # Get text blocks for visualization - mark ALL text blocks for debugging
    text_blocks = page.get_text("blocks")
    
    # For PyMuPDF4LLM debug, let's mark all substantial text blocks
    # regardless of content matching, since PyMuPDF4LLM processes the entire document
    substantial_blocks = []
    
    for block in text_blocks:
        x0, y0, x1, y1, text, *_ = block
        text = text.strip()
        if text and len(text) > 5:  # Lower threshold to catch more blocks
            substantial_blocks.append({
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "text": text
            })
    
    # Sort blocks by position for proper order labeling
    substantial_blocks.sort(key=lambda b: (b["y0"], b["x0"]))
    
    # Draw all substantial block boundaries in orange
    for order, block in enumerate(substantial_blocks, start=1):
        rect = pymupdf.Rect(block["x0"], block["y0"], block["x1"], block["y1"])
        debug_page.draw_rect(rect, color=pymupdf.utils.getColor("orange"), width=1)
        
        # Add order label
        text_point = pymupdf.Point(block["x0"] + 2, block["y0"] - 5)
        debug_page.insert_text(text_point, str(order), fontsize=6, 
                              color=pymupdf.utils.getColor("orange"))
    
    # Add title with info about marked blocks
    title_point = pymupdf.Point(10, 30)
    debug_page.insert_text(title_point, f"PyMuPDF4LLM - Page {page_num} ({len(substantial_blocks)} blocks)", 
                          fontsize=10, color=pymupdf.utils.getColor("orange"))
    
    # Add info about PyMuPDF4LLM processing
    info_text = f"Markdown length: {len(markdown_content)} chars"
    info_point = pymupdf.Point(10, 50)
    debug_page.insert_text(info_point, info_text, fontsize=8, 
                          color=pymupdf.utils.getColor("orange"))
    
    debug_doc.save(output_path)
    debug_doc.close()


def extract_document_pymupdf4llm_with_debug(filepath, debug_output=None):
    """
    Extract text from document using PyMuPDF4LLM for RAG-optimized processing.
    PyMuPDF4LLM provides intelligent text extraction with automatic layout analysis.
    """
    doc = pymupdf.open(filepath)
    all_paragraphs = []
    
    print("Processing with PyMuPDF4LLM (RAG-optimized extraction)...")
    
    try:
        # Convert entire document to markdown using PyMuPDF4LLM
        pages_data = pymupdf4llm.to_markdown(doc)
        
        if pages_data:
            print(f"PyMuPDF4LLM processed {len(pages_data)} pages")
            
            # Process the markdown content
            markdown_content = ""
            page_mappings = {}
            
            # Extract content from all pages and create page mappings
            para_counter = 0
            for page_idx, page_data in enumerate(pages_data):
                if hasattr(page_data, 'text'):
                    text = page_data.text
                else:
                    text = str(page_data)
                
                if text.strip():
                    markdown_content += text + "\n\n"
                    
                    # Map paragraph positions to page numbers
                    sections = re.split(r'\n\s*\n', text)
                    for section in sections:
                        if section.strip():
                            para_start = para_counter
                            para_end = para_counter + len(section.split('\n')) - 1
                            for p in range(para_start, para_end + 1):
                                page_mappings[p] = page_idx + 1
                            para_counter = para_end + 1
            
            # Convert to our standard format
            paragraphs = extract_text_from_markdown(markdown_content, page_mappings)
            all_paragraphs.extend(paragraphs)
            
            # Create debug visualization for first 5 pages
            if debug_output:
                for page_num in range(1, min(6, len(doc) + 1)):  # Pages 1-5 or less if document is shorter
                    debug_path = f"{debug_output}_page_{page_num}.pdf"
                    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                    create_debug_pdf_llm(doc, page_num, markdown_content, debug_path)
                    print(f"Debug PDF saved: {debug_path} (PyMuPDF4LLM extraction)")
            
        else:
            print("No content extracted by PyMuPDF4LLM")
            
    except Exception as e:
        print(f"Error with PyMuPDF4LLM: {e}")
        print("Falling back to PyMuPDF basic extraction...")
        
        # Fallback to basic PyMuPDF extraction
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                all_paragraphs.append({"page": page_num, "text": text.strip()})
    
    doc.close()
    return all_paragraphs


def extract_document_pymupdf4llm_simple(filepath):
    """
    Simple extraction using PyMuPDF4LLM (no debug).
    """
    doc = pymupdf.open(filepath)
    all_paragraphs = []
    
    try:
        # Convert entire document to markdown using PyMuPDF4LLM
        pages_data = pymupdf4llm.to_markdown(doc)
        
        if pages_data:
            # Process the markdown content
            markdown_content = ""
            page_mappings = {}
            
            # Extract content from all pages and create page mappings
            para_counter = 0
            for page_idx, page_data in enumerate(pages_data):
                if hasattr(page_data, 'text'):
                    text = page_data.text
                else:
                    text = str(page_data)
                
                if text.strip():
                    markdown_content += text + "\n\n"
                    
                    # Map paragraph positions to page numbers
                    sections = re.split(r'\n\s*\n', text)
                    for section in sections:
                        if section.strip():
                            para_start = para_counter
                            para_end = para_counter + len(section.split('\n')) - 1
                            for p in range(para_start, para_end + 1):
                                page_mappings[p] = page_idx + 1
                            para_counter = para_end + 1
            
            # Convert to our standard format
            paragraphs = extract_text_from_markdown(markdown_content, page_mappings)
            all_paragraphs.extend(paragraphs)
            
    except Exception as e:
        print(f"Error with PyMuPDF4LLM: {e}")
        print("Falling back to PyMuPDF basic extraction...")
        
        # Fallback to basic PyMuPDF extraction
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                all_paragraphs.append({"page": page_num, "text": text.strip()})
    
    doc.close()
    return all_paragraphs


# Example usage
if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Build paths relative to project root
    pdf_file = script_dir.parent.parent / "data" / "raw" / "ch1_ch14_Brain_and_behavior.pdf"
    debug_output_dir = script_dir.parent.parent / "data" / "debug" / "pdf_extract_pymupdf4llm"
    debug_output_file = debug_output_dir / "pdf_extract_pymupdf4llm"
    
    print(f"Processing PDF with PyMuPDF4LLM: {pdf_file}")
    
    # Test with debug visualization
    result = extract_document_pymupdf4llm_with_debug(
        str(pdf_file), 
        debug_output=str(debug_output_file)
    )
    
    print(f"\nPyMuPDF4LLM extracted {len(result)} paragraphs")
    print("\nFirst 10 paragraphs:")
    for item in result[:10]:
        print(f"[Page {item['page']}] {item['text'][:150]}...")
