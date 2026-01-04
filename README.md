# RAGLab

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![Weaviate](https://img.shields.io/badge/weaviate-vector_db-00C29A)
![Neo4j](https://img.shields.io/badge/neo4j-graph_db-4581C3?logo=neo4j&logoColor=white)
![OpenRouter](https://img.shields.io/badge/openrouter-LLM_gateway-6366F1)
![mxbai-rerank](https://img.shields.io/badge/mxbai--rerank-reranking-FFD21E?logo=huggingface&logoColor=black)
![RAG](https://img.shields.io/badge/RAG-pipeline-purple)
![Built with Claude Code](https://img.shields.io/badge/built_with-Claude_Code-CC785C?logo=anthropic&logoColor=white)

This is an investigation project started to test concepts learned in [DeepLearning.AI course about RAG](https://www.deeplearning.ai/courses/retrieval-augmented-generation-rag/) applying them to an idea I had in mind after reading the fantastic book [Brain and Behaviour, by David Eagleman and Jonathan Downar](https://eagleman.com/books/brain-and-behavior/), which I discovered thanks to  an [Andrej Karpathy talk in youtube](https://youtu.be/fqVLjtvWgq8).

I love also practical philosophy books about wisdom of life from Stoics authors, Schopenhauer, and confucianism and had the idea to get the best of both worlds relating human traits, tendencies and usual struggles worrying some schools of thought with the brain internal functioning, to understand the underlying why to some of the most intriging human behaviour to me.

I started with a simple RAG system with naive chunking and semantic search over my dataset of 19 books (some about neuroscience and some about philosophy), just to soon be aware how difficult it is to get good answers to broad open questions using a RAG simple system, even more difficult mixing two distinct fields of knowledge, one more abstract and another more technical.

So trying to improve the RAG system performance I ended up building a customized evaluation framework to test some of the recent improvements in RAG techniques. I created an user interface to easily tune (embedding collection, preprocessing technique, type of search) and inspect each step result (chunks retrieved, LLM call and responses and final answer) and compare results with different configurations to get an intuition of the effect of each one.

To get more consistent results it runs a comprehensive evaluation using each possible hyperparameter combination (102 cases) over a handcrafted set of test questions that cover both single concept and cross domain concepts. All details are accesible through the links at the end of this README file.

This is custom and simple evaluation framewrok tailored to this specific project and does not aim to be used as a general framework. There are professional frameworks out there for that purpose, but nowadays it is quite easy to construct something like this using the power of coding agents. I did this using Claude Code and Opus 4.5.

I cannot publish the dataset nor database (Weaviate for embeddings, Neo4j from Knowledge Graph) data as the books have intelectual property protection, but I publish the project code and the technical insights and intuitions extracted from my non expert point of view.

---

### Architecture

This are the main components of the application. 


![RAGlab arquitecture](assets/arquitecture.png)

---


### Workflow

The data workflow starts with  books in PDF and follows the standard RAG pipeline. It allows to test different RAG improvement techniques both alone and combined to see the effect of each possible combination chunking strategy/query preprocessing and search type.


![RAGlab workflow](assets/workflow.png)

---

### Corpus

| Book | Author | Category | Tokens |
|------|--------|----------|--------|
| Cognitive Neuroscience: The Biology of the Mind | Michael Gazzaniga | Neuroscience | 455,433 |
| Brain and Behavior | David Eagleman, Jonathan Downar | Neuroscience | 370,663 |
| Biopsychology | John Pinel, Steven Barnes | Neuroscience | 326,159 |
| Behave | Robert M. Sapolsky | Neuroscience | 276,948 |
| Psychobiology of Behaviour | K. Fountoulakis, I. Nimatoudis | Neuroscience | 197,404 |
| Determined | Robert M. Sapolsky | Neuroscience | 194,134 |
| Fundamentals of Cognitive Neuroscience | Nicole M. Gage, Bernard Baars | Neuroscience | 152,365 |
| Cognitive Biology | Luca Tommasi et al. | Neuroscience | 146,231 |
| Letters from a Stoic | Seneca | Philosophy/Wisdom | 281,487 |
| Thinking Fast and Slow | Daniel Kahneman | Philosophy/Wisdom | 204,286 |
| Essays and Aphorisms | Arthur Schopenhauer | Philosophy/Wisdom | 102,616 |
| The Meditations | Marcus Aurelius | Philosophy/Wisdom | 88,693 |
| The Enchiridion | Epictetus | Philosophy/Wisdom | 88,466 |
| The Analects | Confucius | Philosophy/Wisdom | 77,862 |
| The Pocket Oracle | Baltasar Gracián | Philosophy/Wisdom | 54,819 |
| Counsels and Maxims | Arthur Schopenhauer | Philosophy/Wisdom | 54,649 |
| The Wisdom of Life | Arthur Schopenhauer | Philosophy/Wisdom | 51,641 |
| The Art of Living | Epictetus | Philosophy/Wisdom | 23,660 |
| Tao Te Ching | Lao Tzu | Philosophy/Wisdom | 20,415 |
| **Total** | **19 books** | | **3.17M** |

---

### Evaluation

The Streamlit UI allows to change the configuration: embedding collection from the ones in Weaviate, preprocessing technique applied (HyDE, Query Decomposition, GraphRAG), search type (keyword, hybrid or pure semantic) and if reranking is used or not.

In the UI you can see the chunks retrieved, the score of each chunk, the intermediante LLM interactions for Query Decomposition or HyDE) and the final answer, so in one place you can easily compare intermediate steps and final results of each configuration for same question.
In addition to user direct evaluation at UI, an evaluation stage is included using RAGAS metrics over a set of handcrafted questions combining single concept questions and cross domain questions.


## Techniques Implemented

| Technique | Paper | What It Does |
|-----------|-------|--------------|
| **HyDE** | [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) | Generates hypothetical answers for semantic matching |
| **Query Decomposition** | [arXiv:2507.00355](https://arxiv.org/abs/2507.00355) | Breaks complex questions into sub-queries with RRF merging |
| **Contextual Chunking** | [Anthropic Blog](https://www.anthropic.com/news/contextual-retrieval) | LLM-generated context prepended to chunks (-35% retrieval failures) |
| **RAPTOR** | [arXiv:2401.18059](https://arxiv.org/abs/2401.18059) | Hierarchical summarization tree with UMAP + GMM clustering |
| **GraphRAG** | [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) | Knowledge graph + Leiden communities for cross-document reasoning |
| **GraphRAG Auto-Tuning** | [MS Research](https://www.microsoft.com/en-us/research/blog/graphrag-auto-tuning-provides-rapid-adaptation-to-new-domains/) | Discovers entity types from corpus content (per-book resumable) |

Plus: Hybrid search (BM25 + vector), cross-encoder reranking, structured LLM outputs, and RAGAS evaluation framework.

## Quick Start

```bash
docker compose up -d              # Start Weaviate + Neo4j
streamlit run src/ui/app.py       # Open http://localhost:8501
```

See [Getting Started](docs/getting-started.md) for full pipeline commands.

## Technologies

| Category | Tools |
|----------|-------|
| **Vector Database** | Weaviate (HNSW + BM25 hybrid) |
| **Graph Database** | Neo4j (GDS plugin for Leiden communities) |
| **LLM API** | OpenRouter (GPT-4, Claude, embeddings) |
| **NLP** | spaCy (en_core_sci_sm), tiktoken |
| **PDF Processing** | Docling |
| **Data Validation** | Pydantic (structured LLM outputs) |
| **UI** | Streamlit |
| **Evaluation** | RAGAS framework |
| **Infrastructure** | Docker, Conda |

## Documentation

For implementation details, design decisions, and code walkthroughs:

- **[Getting Started](docs/getting-started.md)** — Installation, prerequisites, commands
- **[Architecture](docs/architecture.md)** — Pipeline diagram, project structure
- **[Content Preparation](docs/content-preparation/)** — PDF extraction, cleaning
- **[Chunking Strategies](docs/chunking/)** — Section, Contextual, RAPTOR
- **[Preprocessing Strategies](docs/preprocessing/)** — HyDE, Decomposition, GraphRAG
- **[Evaluation Framework](docs/evaluation/)** — RAGAS metrics and results

## Key Insights

Building this pipeline taught me that RAG is deceptively complex:

**PDF parsing is harder than expected.** Scientific books with complex layouts, figures, and footnotes break naive extraction. Docling helped, but significant cleaning was still needed.

**Prompts make or break LLM-based techniques.** HyDE, RAPTOR summarization, and entity extraction all depend heavily on prompt engineering. Small wording changes dramatically affect output quality.

**Evaluation is the hardest part.** Generating good test questions for RAGAS requires domain expertise. The gap between "looks reasonable" and "measurably good" is where real learning happens.

**GraphRAG complexity is justified.** The knowledge graph + Leiden communities approach seemed over-engineered at first, but it handles cross-document reasoning that vector search alone cannot.

## License

MIT
