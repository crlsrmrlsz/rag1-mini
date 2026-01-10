# PDF extraction tools for scientific RAG pipelines: A 2026 comparison

For extracting complex scientific documents into RAG systems, **MinerU**, **Docling**, and **Marker** lead open-source options, while **Mathpix** dominates equation-heavy content and **Azure AI Document Intelligence** offers the best cloud-based LaTeX formula extraction. No single tool handles all scientific document challenges—the optimal approach combines specialized tools: use GROBID for citations, Mathpix for equations, and Docling or MinerU for general layout understanding.

The landscape has shifted dramatically since 2024. Vision-Language Models now power the most capable parsers, though NVIDIA's benchmarks show specialized OCR pipelines still achieve **32x higher throughput** than VLM approaches. Community consensus has moved away from basic tools like pdfplumber toward ML-based alternatives, with **pymupdf4llm** emerging as the speed/quality sweet spot at 0.12 seconds per page.

---

## Open-source tools: Traditional rule-based parsers

Traditional parsers excel at speed and simplicity but struggle with complex scientific layouts. These tools work directly on PDF structure without ML models.

| Tool | GitHub Stars | Downloads/Week | License | Speed | Best For |
|------|-------------|----------------|---------|-------|----------|
| **PyMuPDF** | 8,800 | 2M+ | AGPL-3.0 | 0.1s/page | High-speed batch processing |
| **pypdf** | 9,700 | 2M+ | BSD-3 | 3.5s/page | PDF manipulation, 96% text accuracy |
| **pdfplumber** | 9,400 | 1.8M | MIT | 9.5s/page | Table extraction, debugging |
| **PDFMiner** | 6,800 | 1.1M | MIT | 5.8s/page | Layout-faithful extraction |
| **Camelot** | 3,600 | 500K | MIT | Variable | Best-in-class table extraction |
| **Tabula** | 2,300 | 200K | MIT | Variable | Quick table extraction |

**PyMuPDF** stands out as the fastest option with pymupdf4llm extension providing LLM-ready Markdown output. It handles multi-column layouts reasonably well and extracts images with full metadata. The **AGPL license** requires code sharing for public deployments, though commercial licenses are available.

**Camelot** achieves 99%+ accuracy on tables with visible borders using Lattice mode and whitespace-based Stream mode for borderless tables. It provides quality metrics for validation and exports to pandas DataFrames. However, it requires Ghostscript and cannot process scanned documents.

**Key limitations across all traditional tools**: No semantic understanding of equations (extracted as symbols only), no built-in citation parsing, and multi-column layouts often require manual configuration or post-processing.

---

## Open-source tools: ML and vision-based parsers

ML-based parsers represent the current state-of-the-art for scientific document understanding, using transformer architectures and vision models to understand document structure.

| Tool | GitHub Stars | Technology | Scientific Content | Hardware |
|------|-------------|------------|-------------------|----------|
| **MinerU** | 48,800 | DocLayout-YOLO, PaddleOCR | ⭐⭐⭐⭐⭐ | GPU recommended |
| **Docling** | 37,000+ | DocLayNet, TableFormer | ⭐⭐⭐⭐ | CPU optimized |
| **Marker** | 30,800 | Surya, Donut | ⭐⭐⭐⭐ | GPU recommended |
| **Surya** | 19,000 | EfficientViT, Donut | ⭐⭐⭐ | GPU recommended |
| **Unstructured** | 13,200 | Detectron2, YOLOX | ⭐⭐⭐ | CPU only (OSS) |
| **Nougat** | 9,700 | Swin Transformer, mBART | ⭐⭐⭐⭐⭐ | GPU required |
| **GOT-OCR 2.0** | 7,000+ | Vision-Language Model | ⭐⭐⭐⭐⭐ | GPU required |

**MinerU** (OpenDataLab) has become the most popular choice with **48,800 GitHub stars**. Built from InternLM's pre-training pipeline, it handles 109 languages via PaddleOCR and achieves excellent cross-page paragraph concatenation. Version 2.5 introduced a decoupled VLM for high-resolution parsing. The **AGPL-3.0 license** requires compliance for commercial use.

**Docling** (IBM Research, MIT licensed) joined the Linux Foundation AI & Data project and offers the most enterprise-friendly option. Its TableFormer model achieves **97.9% table structure accuracy** in benchmarks. Native integrations exist with LangChain, LlamaIndex, Haystack, and Crew AI. Processing runs at ~2.45 pages/second on MacBook Pro M3 Max without requiring a GPU.

**Nougat** (Meta AI) remains the gold standard for arXiv-style papers, producing LaTeX output directly from rasterized pages without OCR. It handles scanned papers well but operates 4x slower than Marker. The **CC-BY-NC model weights** restrict commercial use.

**GOT-OCR 2.0** represents the unified "OCR 2.0" paradigm—a single 580M parameter VLM handling text, tables, math formulas, molecular formulas, charts, and even sheet music. Apache 2.0 licensed with **1M+ HuggingFace downloads**.

---

## Scientific-focused tools: GROBID and specialized parsers

**GROBID** (GeneRation Of BIbliographic Data) has processed scientific literature at scale for over 15 years, powering Semantic Scholar, ResearchGate, and Internet Archive Scholar.

**Architecture**: Cascade of sequence labeling models (CRF by default, BiLSTM-CRF or BERT-CRF via DeLFT). Uses "Layout Tokens" that incorporate visual positioning information alongside text.

**Capabilities**: 
- Reference parsing: **~0.90 F1** on bioRxiv using deep learning models
- 68 fine-grained structural labels (paragraphs, section titles, footnotes, figures, tables)
- PDF coordinate extraction for all identified structures
- Patent processing, funding extraction, date normalization

**Throughput**: 915,000+ PDFs/day possible in production deployments.

**Limitations**: Struggles with heavily scanned documents, limited equation/formula sophistication, primarily optimized for Latin scripts.

**Specialized scientific modules** extend GROBID: grobid-quantities (physical measurements), grobid-superconductors (materials science), datastet (dataset identification), and software-mention (software citation recognition).

### Chemistry-specific tools

| Tool | Capability | Output Format |
|------|-----------|---------------|
| **Mathpix** | 99%+ chemical formula accuracy | SMILES, ChemDraw |
| **DECIMER.ai** | Deep learning structure recognition | SMILES |
| **ChemScraper** | Born-digital PDF graphics | 98.4% USPTO accuracy |
| **MolScribe** | Image-to-graph molecular structures | Graph representation |

### Math and physics equation tools

**Mathpix** dominates equation recognition with **99%+ accuracy** on PhD-level mathematics. **pix2tex** (LaTeX-OCR) provides an open-source alternative using Vision Transformers. **Surya** now includes the deprecated Texify functionality via `surya_latex_ocr`.

---

## Commercial cloud services compared

Cloud services offer scalability and managed infrastructure but vary significantly in scientific document capabilities.

| Service | Formula Support | Table Quality | Price (per 1K pages) | Strengths |
|---------|----------------|---------------|---------------------|-----------|
| **Azure AI Document Intelligence** | ✅ LaTeX output | ⭐⭐⭐⭐ | $1.50 + $6 add-ons | Markdown output, custom training |
| **Google Document AI** | ✅ Add-on | ⭐⭐⭐⭐ | $1.50 + $6 add-ons | 200+ languages, Gemini-powered |
| **Adobe PDF Services** | ❌ | ⭐⭐⭐⭐⭐ | Enterprise only (~$25K/yr) | Best table extraction |
| **Amazon Textract** | ❌ | ⭐⭐⭐⭐ | $1.50-$70 | AWS integration, query feature |

**Azure AI Document Intelligence** (v4.0, November 2024) uniquely provides **LaTeX formula output** with bounding polygons—critical for scientific papers. The $6/1,000 pages formula add-on enables extraction alongside standard OCR. Markdown output format integrates directly with LLM pipelines.

**Google Document AI** offers formula recognition via Enterprise OCR add-on and leads in language support (200+ languages). The Gemini 2.5 Flash-powered custom extractors can be fine-tuned with as few as 10 documents.

**Adobe PDF Services** benchmarked as **best for table extraction** from academic documents in 2023 studies but lacks equation recognition and requires enterprise agreements (reportedly $25,000/year minimum).

**Key finding**: All cloud services struggle with complex scientific documents according to academic benchmarks. Hybrid approaches combining cloud services for general structure with specialized tools for equations and citations produce the best results.

---

## Commercial specialized tools for scientific content

| Tool | Focus | Equation Quality | Pricing | RAG Optimization |
|------|-------|-----------------|---------|------------------|
| **Mathpix** | STEM documents | ⭐⭐⭐⭐⭐ | $0.0035/page | Good |
| **Reducto** | RAG pipelines | ⭐⭐⭐ | $0.015/credit | ⭐⭐⭐⭐⭐ |
| **ABBYY** | Enterprise OCR | ⭐⭐ | $199-$399+ | Limited |
| **Kofax** | Batch scanning | ⭐ | $579-$17K+ | Limited |

**Mathpix** serves 3+ million users and has converted **5+ billion PDF pages**. Its SuperNet-100 neural networks handle handwritten equations, chemistry diagrams (SMILES output), and two-column scientific layouts. Pro plan: $4.99/month for 1,000 PDF pages and 5,000 images.

**Reducto** (Y Combinator, $108M raised through 2025) built specifically for RAG pipelines with "Agentic OCR"—a multi-pass self-correcting framework. **Bounding box citations** for every extracted element enable provenance tracking. Layout-aware semantic chunking preserves meaning across chunk boundaries. Used by Scale AI and Fortune 10 enterprises.

**LlamaParse** (LlamaIndex) provides GenAI-native parsing with VLM-powered OCR. Free tier: 1,000 pages/day. Paid tiers range from $0.003/page (cost-effective) to $0.09/page (Sonnet 4.0 agentic mode). Has processed **500M+ documents**.

---

## Technology architectures explained

Understanding the underlying technology helps predict tool behavior on different document types.

**Rule-based parsing** (PyMuPDF, pdfplumber, PDFMiner): Extract text by reading PDF object coordinates directly. Fast but miss semantic structure. Multi-column handling requires manual tuning.

**Transformer-based document understanding** (Nougat, GOT-OCR, Docling): Encode document images through vision transformers (Swin, EfficientViT) and decode structure through language models. Handle complex layouts but require GPU and run slower.

**Hybrid approaches** (MinerU, Marker, Unstructured): Combine object detection (YOLO, Detectron2) for layout analysis with OCR engines (PaddleOCR, Tesseract, Surya) for text recognition. Balance speed and quality.

**Vision-Language Models** (LlamaParse, GOT-OCR, Qwen2.5-VL): Process visual and textual content simultaneously through unified architectures. Best semantic understanding but highest computational cost.

**NVIDIA's 2024-2025 benchmarks** found specialized OCR pipelines outperform VLMs for retrieval tasks by **7.2% on Recall@5** while achieving 32x higher throughput on single A100 GPUs. VLMs excel at answer generation from complex visuals but struggle with hallucination and incomplete extraction.

---

## RAG pipeline recommendations for scientific literature

Extraction quality matters more than model size. Well-chunked content dramatically improves answers even with smaller models.

**Chunking strategies** tested by NVIDIA achieved highest accuracy (**0.648**) with page-level chunking, offering lowest variance. For scientific documents, 300-1,000 tokens per chunk with 10-20% overlap works well. Preserve structure (titles, headings, captions) as metadata rather than inline text.

**Handling figures and tables**: Markdown format has become standard for table output (Docling, pymupdf4llm). For figures, PaperQA2's approach creates `ParsedMedia` objects with one-to-many chunk relationships, using LLM-generated synthetic captions for embedding without polluting source text.

**Citation-aware RAG**: PaperQA2 demonstrates superhuman performance on scientific tasks through automatic metadata fetching from Crossref, Semantic Scholar, and Unpaywall. It includes citation counts, retraction checks, and journal quality data in retrieval ranking.

**Recommended tool combinations by use case**:

- **High-volume scientific papers**: MinerU (layout) + GROBID (citations) + Mathpix API (equations)
- **Enterprise RAG with compliance**: Docling (MIT licensed, CPU-capable) + Azure Document Intelligence (formulas)
- **Academic research projects**: Nougat (LaTeX output) + PaperQA2 (agentic RAG)
- **Speed-critical pipelines**: pymupdf4llm (0.12s/page) with Camelot fallback for tables
- **Chemistry documents**: Mathpix (SMILES) + DECIMER.ai (structure recognition)

---

## Comprehensive tool comparison matrix

| Tool | Stars | License | Layout | Tables | Equations | Images | Speed | Price |
|------|-------|---------|--------|--------|-----------|--------|-------|-------|
| **MinerU** | 48.8K | AGPL | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | Free |
| **Docling** | 37K | MIT | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 2.45p/s | Free |
| **Marker** | 30.8K | GPL | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Fast | Free/Paid |
| **Nougat** | 9.7K | CC-BY-NC | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Slow | Free (NC) |
| **GOT-OCR** | 7K | Apache | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | Free |
| **PyMuPDF** | 8.8K | AGPL | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | 0.1s/p | Free |
| **GROBID** | 4.5K | Apache | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | Fast | Free |
| **Mathpix** | N/A | Proprietary | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Fast | $0.0035/p |
| **LlamaParse** | N/A | Proprietary | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ~45s/block | $0.003-0.09/p |
| **Reducto** | N/A | Proprietary | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Fast | $0.015/credit |
| **Azure Doc Intel** | N/A | Cloud | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Fast | $1.50+/1K |
| **Google Doc AI** | N/A | Cloud | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Fast | $1.50+/1K |

---

## Conclusion

The optimal PDF extraction strategy for scientific RAG systems in 2026 combines multiple specialized tools rather than relying on any single solution. **MinerU** and **Docling** lead open-source general-purpose extraction with strong table handling. **GROBID** remains unmatched for citation and reference parsing with 0.90 F1 scores. **Mathpix** dominates equation recognition at $0.0035/page—far cheaper than manual conversion.

For production systems, consider a **tiered approach**: fast tools like pymupdf4llm (0.12s/page) for simple documents, falling back to sophisticated parsers like Docling or LlamaParse for complex layouts. The **page-level chunking strategy** achieves highest RAG accuracy with lowest variance in NVIDIA's benchmarks.

The most significant 2024-2025 development is the emergence of **Vision-Language Model approaches** (GOT-OCR, GraniteDocling) that handle text, tables, equations, and figures through unified architectures—pointing toward simpler pipelines as these models mature. However, specialized OCR pipelines still outperform VLMs for retrieval tasks, making hybrid approaches the pragmatic choice today.