from docling.document_converter import DocumentConverter
from pathlib import Path

# Define paths
project_root = Path(__file__).parent.parent.parent
input_file = project_root / "data" / "raw" / "neuroscience" / "Brain_and_behavior_a_cognitive_neuroscience_perspective_David_Eagleman_Jonathan_Downar.pdf"
output_dir = project_root / "data" / "processed_docling"
output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

# Derive output filename: replace .pdf with .md
output_file = output_dir / (input_file.stem + ".md")

# Convert document
converter = DocumentConverter()
result = converter.convert(input_file)

# Save Markdown to file
output_file.write_text(result.document.export_to_markdown(), encoding="utf-8")

print(f"Markdown saved to: {output_file}")



