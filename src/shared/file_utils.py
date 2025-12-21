import logging
from pathlib import Path
from typing import List

def setup_logging(name: str) -> logging.Logger:
    """Configures a standard logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(name)

def get_file_list(source_dir: Path, extension: str) -> List[Path]:
    """Recursively finds all files with specific extension."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    return list(source_dir.rglob(f"*.{extension}"))

def get_output_path(source_path: Path, source_root: Path, output_root: Path, new_extension: str = None) -> Path:
    """
    Calculates the mirror output path.
    Example: raw/neuroscience/book.pdf -> processed/neuroscience/book.md
    """
    relative_path = source_path.relative_to(source_root)
    destination = output_root / relative_path
    
    if new_extension:
        destination = destination.with_suffix(new_extension)
        
    return destination
