import re
from src.config import LINE_ARTIFACT_PATTERNS, INLINE_REMOVAL_PATTERNS

def clean_whole_lines(text: str) -> str:
    """Removes lines that are likely captions or artifacts."""
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        skip = False
        for pattern, replacement in LINE_ARTIFACT_PATTERNS:
            if re.search(pattern, line, flags=re.IGNORECASE):
                skip = True
                break
        if not skip:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def fix_broken_chapters(text: str) -> str:
    """
    Fixes broken chapter headers (Multi-line -> Single line).
    """
    # Pattern 1: Numbered Chapters
    p1 = r'(?m)^\s*CHAPTER\s*$\n+\s*^##\s*(.+?)\s*$\n+\s*^(\d+)\s*$'
    text = re.sub(p1, r'# CHAPTER \2 \1', text)
    
    # Pattern 2: Unnumbered/Intro Chapters
    p2 = r'(?m)^\s*##\s*CHAPTER\s*$\n+\s*^##\s*(.+?)\s*$'
    text = re.sub(p2, r'# CHAPTER \1', text)
    return text

def clean_inline_formatting(text: str) -> str:
    """Refines string spacing and removes specific artifacts."""
    text = text.replace('/u2014.d', ' - ')
    text = text.replace('/u2014', ' - ')
    
    for pattern in INLINE_REMOVAL_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Fix 'mu- opioid' -> 'mu-opioid'
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1-\2', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def consolidate_broken_paragraphs(paragraphs: list) -> list:
    """
    Merges paragraphs split by formatting errors.
    Logic: If P1 doesn't end in punctuation and P2 starts lowercase -> Merge.
    """
    if not paragraphs:
        return []
    
    merged = []
    buffer = paragraphs[0].strip()
    
    for i in range(1, len(paragraphs)):
        current_p = paragraphs[i].strip()
        if not current_p:
            continue
            
        ends_with_terminal = buffer.endswith(('.', '!', '?', ':', ';', '"', '”'))
        starts_lowercase = current_p[0].islower() if len(current_p) > 0 else False
        ends_connector = buffer.endswith((',', '-'))

        if (not ends_with_terminal) or starts_lowercase or ends_connector:
            buffer += " " + current_p
        else:
            merged.append(buffer)
            buffer = current_p
            
    if buffer:
        merged.append(buffer)
        
    return merged

def run_structural_cleaning(md_content: str) -> str:
    """Orchestrates Phase 3 Cleaning."""
    # 1. Fix Headers first
    md_content = fix_broken_chapters(md_content)
    
    # 2. Remove Artifact Lines
    clean_content = clean_whole_lines(md_content)
    
    # 3. Split by Headers (#) to preserve structure
    sections = re.split(r'(^#+\s+.*$)', clean_content, flags=re.MULTILINE)
    
    reconstructed_text = []
    
    for segment in sections:
        segment = segment.strip()
        if not segment: continue
            
        # Keep Headers as is
        if segment.startswith('#'):
            reconstructed_text.append(f"\n\n{segment}\n\n")
            continue
            
        # Process Body Text
        raw_paragraphs = segment.split('\n\n')
        stitched_paragraphs = consolidate_broken_paragraphs(raw_paragraphs)
        
        for p in stitched_paragraphs:
            reflowed = p.replace('\n', ' ')
            final_p = clean_inline_formatting(reflowed)
            
            if len(final_p) > 0:
                reconstructed_text.append(f"{final_p}\n\n")
                
    return "".join(reconstructed_text)