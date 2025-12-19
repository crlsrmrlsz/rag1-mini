# Project Progress - RAG1 Mini

## Current Status: Stages 1-7C Complete ✅

**Last Updated:** December 19, 2025

### ✅ Completed Stages

#### **Stage 1: PDF Extraction** ✅
- **Status:** COMPLETE
- **Input:** 19 PDF files from `raw/` directory
- **Output:** Raw markdown files in `processed/01_raw_extraction/`
- **Books Processed:**
  - **Neuroscience (8 books):** Sapolsky, Pinel & Barnes, Eagleman & Downar, Gazzaniga, Tommasi et al., Sapolsky (Determined), Gage & Bernard, Fountoulakis & Nimatoudis
  - **Wisdom (11 books):** Kahneman, Schopenhauer (multiple works), Lao Tzu, Seneca, Confucius, Epictetus (multiple works), Marcus Aurelius, Baltasar Gracian

#### **Stage 2: Processing & Cleaning** ✅  
- **Status:** COMPLETE
- **Components:**
  - Manual review in `processed/02_manual_review/`
  - Markdown cleaning in `processed/03_markdown_cleaning/`
  - All files successfully cleaned and standardized

#### **Stage 3: NLP Segmentation & Chunking** ✅
- **Status:** COMPLETE  
- **Output Location:** `processed/04_nlp_chunks/`
- **Output Format:** JSON files for each book
- **Chunking Details:**
  - **Neuroscience chunks:** 8 books with full NLP processing
  - **Wisdom chunks:** 11 books with full NLP processing
  - **Total outputs:** 19 JSON files with structured paragraphs and sentences
  - **Metadata:** Includes context, sentences, num_sentences for each paragraph

#### **Stage 4: Section-Based Chunking** ✅
- **Status:** COMPLETE
- **Output:** `processed/05_final_chunks/section/` (6,245 chunks)
- **Config:** 800 tokens max, 2-sentence overlap

#### **Stage 5: Embedding Generation** ✅
- **Status:** COMPLETE
- **Output:** `processed/06_embeddings/` (19 JSON files)
- **Model:** text-embedding-3-large (3072 dimensions) via OpenRouter
- **Batch processing:** 12,000 token batches with retry logic

#### **Stage 6: Weaviate Vector Storage** ✅
- **Status:** COMPLETE
- **Collection:** `RAG_section800_embed3large_v1`
- **Index:** HNSW with cosine distance
- **Total objects:** 6,249 chunks uploaded

#### **Stage 7A: Query Layer** ✅
- **Status:** COMPLETE (December 19, 2025)
- **New file:** `src/vector_db/weaviate_query.py`
- **Functions:** `query_similar()`, `query_hybrid()`, `list_available_books()`
- **Features:** Book filtering, vector search, hybrid search (vector + BM25)

#### **Stage 7B: Streamlit UI** ✅
- **Status:** COMPLETE (December 19, 2025)
- **New files:** `src/ui/app.py`, `src/ui/services/search.py`
- **Run:** `streamlit run src/ui/app.py`
- **Features:** Book selection, search type toggle, collection selector

#### **Stage 7C: RAGAS Evaluation** ✅
- **Status:** COMPLETE (December 19, 2025)
- **New module:** `src/evaluation/` with `ragas_evaluator.py`
- **Run:** `python -m src.run_stage_7_evaluation`
- **Features:**
  - OpenRouter chat integration for answer generation
  - RAGAS metrics: faithfulness, answer relevancy, context precision
  - 10 curated test questions (single concept, cross-domain, open-ended)
  - JSON evaluation reports with per-question and aggregate scores
- **Initial Results:**
  - Faithfulness: 1.0 (answers grounded in context)
  - Answer Relevancy: 0.96-1.0 (answers address questions)

### 📊 Processing Statistics

**Data Flow:**
```
raw/ (19 PDFs) 
    ↓ [Stage 1: Extraction]
processed/01_raw_extraction/ (19 MD files)
    ↓ [Stage 2: Review & Cleaning] 
processed/02_manual_review/ + processed/03_markdown_cleaning/ (19 MD files)
    ↓ [Stage 3: NLP Segmentation]
processed/04_nlp_chunks/ (19 JSON files with structured paragraphs)
    ↓ [Stage 4: Section Chunking]
processed/05_final_chunks/section/ (19 JSON files with 6,245 total chunks)
```

**Content Categories:**
- **Neuroscience:** 8 books (psychology, cognitive science, brain research)
- **Wisdom:** 11 books (philosophy, stoicism, eastern philosophy, behavioral economics)

### 🔧 Technical Implementation

**Core Scripts:**
- `run_stage_1_extraction.py` - PDF to markdown extraction
- `run_stage_2_processing.py` - Cleaning and manual review processing  
- `run_stage_3_segmentation.py` - NLP-based paragraph and sentence segmentation
- `run_stage_4_chunking.py` - Section-aware chunking with overlap **[UPDATED]**

**Processors:**
- `text_cleaner.py` - Markdown cleaning and standardization
- `nlp_segmenter.py` - Advanced NLP for semantic segmentation (spaCy-based)
- `naive_chunker.py` - Sequential chunking with overlap **[UPDATED]**

**Configuration:**
- `config.py` - Centralized settings for chunk sizes, paths, spaCy model
- **Recent Fixes:** Updated spaCy model from `en_core_sci_sm` to `en_core_web_sm`

**Dependencies Installed:**
- `tiktoken` - Token counting for OpenAI-compatible models
- `spacy` - NLP processing framework
- `en_core_web_sm` - English language model for spaCy

### 📁 Current Directory Structure

```
data/
├── raw/                    # Original PDFs (19 files)
├── processed/
│   ├── 01_raw_extraction/   # Stage 1 output (19 MD files)
│   ├── 02_manual_review/    # Stage 2 review (19 MD files)
│   ├── 03_markdown_cleaning/ # Stage 2 cleaned (19 MD files)
│   ├── 04_nlp_chunks/       # Stage 3 output (19 JSON files)
│   ├── 05_final_chunks/     # Stage 4 output (19 JSON files, 6,245 chunks)
│   └── 06_embeddings/       # Stage 5 output (19 JSON files with embeddings)
├── evaluation/
│   ├── test_questions.json  # Stage 7C test questions with ground truth
│   └── results/             # RAGAS evaluation reports
└── logs/
    └── cleaning_report.log
```

### 🎯 Current: Run Full RAGAS Evaluation

**Selected Models (Quality Tier, ~$0.50 for 10 questions):**
```bash
python -m src.run_stage_7_evaluation \
  --generation-model google/gemini-3-flash \
  --evaluation-model anthropic/claude-sonnet-4.5
```

**See:** `memory-bank/model-selection.md` for full pricing and alternatives.

---

### 🔮 Stage 8: Advanced RAG Improvements (Planned)

**Why improve?** Current naive chunking loses context for scientific texts. Example:
> "This receptor shows increased activity..." — What receptor? Which brain region?

#### 1. Contextual Retrieval (Anthropic, Sept 2024)
- **What:** Prepend LLM-generated context to each chunk before embedding
- **Why:** 67% reduction in retrieval failure (with BM25 + reranking)
- **Cost:** ~$3 to contextualize all 6,245 chunks with prompt caching
- **Result:** `RAG_contextual_embed3large_v1` collection

#### 2. Semantic Chunking
- **What:** Chunk based on semantic similarity, not fixed size
- **Why:** Variable-size chunks preserve complete concepts
- **Best for:** Philosophy texts with varying density

#### 3. Increased Top-K + Reranking
- **What:** Retrieve 20 chunks, LLM reranks to top 5
- **Why:** Reduces noise, improves faithfulness for complex questions

#### 4. Hybrid Search Tuning
- **What:** Test alpha values (0.3, 0.5, 0.7) for vector/keyword balance
- **Why:** Technical terms benefit from keyword matching

### 📊 Evaluation Workflow

```
1. Run baseline (current) → identify weak points
2. Implement improvement → create new collection
3. Re-evaluate → compare to baseline
4. Document improvement delta
```

### 💡 Key Achievements

1. **Complete Pipeline:** Successfully implemented full extraction → cleaning → NLP segmentation → chunking → embedding → search → evaluation pipeline
2. **Quality Output:** High-quality, structured chunks with proper metadata and context
3. **Section Awareness:** Chunking respects document structure and section boundaries
4. **Overlap Strategy:** 2-sentence overlap ensures context continuity between chunks
5. **Token Optimization:** 800-token chunks optimized for embedding models
6. **Category Organization:** Clear separation between neuroscience and wisdom content
7. **Comprehensive Testing:** All import errors resolved, all stages functional
8. **Scalable Architecture:** Modular design ready for RAG integration
9. **RAGAS Evaluation:** Full evaluation framework with OpenRouter integration for quality measurement

### 📝 Recent Updates (December 15, 2025)

**Import Error Resolution:**
- Fixed missing exports in `src/utils/__init__.py`
- Installed required dependencies: `tiktoken`, `spacy`
- Updated spaCy model configuration to use `en_core_web_sm`
- Resolved all import conflicts across stages 3 and 4

**Stage 4 Validation:**
- Successfully processed all 19 books through section chunking
- Generated 6,245 high-quality chunks with proper overlap
- Verified chunk quality and token count compliance
- All output files properly formatted and structured

### 📝 Notes

- All books successfully processed through all 4 stages
- No failed extractions or processing errors
- Clean, consistent file naming convention
- All import dependencies resolved and functional
- Ready for vector embedding and database integration
- Comprehensive logging and error handling implemented
