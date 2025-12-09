import spacy
import re
from typing import List, Dict, Tuple
from src.config import SPACY_MODEL, VALID_ENDINGS
from src.utils import setup_logging

logger = setup_logging(__name__)

class SemanticSegmenter:
    def __init__(self):
        logger.info(f"Loading SciSpaCy model: {SPACY_MODEL}...")
        try:
            self.nlp = spacy.load(SPACY_MODEL, disable=["ner"])
        except OSError:
            raise OSError(f"Model '{SPACY_MODEL}' not found. Please install it.")

    def _get_sentences(self, text: str) -> List[str]:
        """Uses SciSpaCy to intelligently split sentences."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def _filter_sentences(self, sentences: List[str]) -> Tuple[List[str], List[str]]:
        """
        Phase 5: Filter out noise (fragments, lowercase starts).
        Returns: (kept_sentences, removed_log)
       
        """
        kept = []
        removed = []
        
        for sent in sentences:
            reason = None
            if len(sent.split()) < 2:
                reason = "Too short"
            elif sent and sent[0].islower():
                reason = "Starts lowercase"
            elif sent and not sent.endswith(VALID_ENDINGS):
                reason = "No terminal punctuation"
            
            if reason:
                removed.append(f"[{reason}] {sent}")
            else:
                kept.append(sent)
                
        return kept, removed

    def process_document(self, clean_text: str, book_name: str) -> List[Dict]:
        """
        Phase 4: Segmentation.
        Splits by header, extracts sentences, and applies Phase 5 filtering.
        """
        sections = re.split(r'(^#+\s+.*$)', clean_text, flags=re.MULTILINE)
        
        processed_chunks = []
        current_chapter = "Unknown Chapter"
        current_section = ""
        
        for segment in sections:
            segment = segment.strip()
            if not segment: continue
            
            # Update Context
            if segment.startswith('#'):
                clean_header = segment.lstrip('#').strip()
                if segment.startswith('# '):
                    current_chapter = clean_header
                    current_section = ""
                elif segment.startswith('##'):
                    current_section = clean_header
                continue
            
            # Process Body Paragraphs
            paragraphs = segment.split('\n\n')
            for p in paragraphs:
                p = p.strip()
                if not p: continue
                
                # 1. NLP Split
                raw_sentences = self._get_sentences(p)
                
                # 2. Filter (Phase 5)
                valid_sentences, _ = self._filter_sentences(raw_sentences)
                
                if not valid_sentences:
                    continue

                # 3. Construct Context String
                context_str = f"{book_name} > {current_chapter}"
                if current_section:
                    context_str += f" > {current_section}"
                
                chunk_data = {
                    "context": context_str,
                    "text": " ".join(valid_sentences), # Reconstructed text
                    "sentences": valid_sentences,
                    "num_sentences": len(valid_sentences)
                }
                processed_chunks.append(chunk_data)
                
        return processed_chunks