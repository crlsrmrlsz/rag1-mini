from pathlib import Path
import fitz             # PyMuPDF (classic binding)
import pymupdf.layout   # Required to enable layout analysis features
import pymupdf4llm      # High-level Markdown extraction utilities

def get_failed_pages(pdf_path: Path):
    """
    Iterates through a PDF page by page.
    Prints specific errors and returns a list of failed page indices.
    """
    failed_pages = []

    try:
        # Open document to get page count
        with fitz.open(pdf_path) as doc:
            print(f"\n--- Processing: {pdf_path.name} ({len(doc)} pages) ---")

            for i in range(len(doc)):
                try:
                    # Attempt to convert a single page
                    # show_progress=False prevents cluttering the console
                    _ = pymupdf4llm.to_markdown(str(pdf_path), pages=[i], show_progress=False)
                except Exception as e:
                    print(f"  [!] Exception on page {i}: {e}")
                    failed_pages.append(i)

    except Exception as e:
        print(f"Could not open file {pdf_path.name}: {e}")
        return

    # Summary for this file
    if failed_pages:
        print(f"  => SUMMARY: {len(failed_pages)} errors found on pages: {failed_pages}")
    else:
        print("  => SUMMARY: No errors found.")

def main():
    """Define project paths and run the extraction pipeline on specific files."""
    
    project_root = Path(__file__).parent.parent
    # Adjust 'neuroscience' path if your files are in 'data/raw' directly
    neuro_dir = project_root / "data" / "raw" / "neuroscience" 

    test_only = {
        "Cognitive_Biology_Evolutionary_and_Developmental_Perspectives_on_Mind_Brain_and_Behavior_Luca_Tommasi_Mary_A._Peterson_Lynn_Nadel.pdf",
        "Brain_and_behavior_a_cognitive_neuroscience_perspective_David_Eagleman_Jonathan_Downar.pdf",
        "Biopsychology_Global_Ed_11th_John_Pinel_Steven_Barnes.pdf",
    }

    for file_name in test_only:
        file_path = neuro_dir / file_name
        
        if file_path.exists():
            get_failed_pages(file_path)
        else:
            print(f"\n[!] File not found: {file_name}")

if __name__ == "__main__":
    main()