# RAGLab

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![Weaviate](https://img.shields.io/badge/weaviate-vector_db-00C29A)
![Neo4j](https://img.shields.io/badge/neo4j-graph_db-4581C3?logo=neo4j&logoColor=white)
![OpenRouter](https://img.shields.io/badge/openrouter-LLM_gateway-6366F1)
![RAGAS](https://img.shields.io/badge/RAGAS-evaluation-09B3AF)
![scispaCy](https://img.shields.io/badge/scispaCy-NLP-1A9CFC?logo=spacy&logoColor=white)
![Docling](https://img.shields.io/badge/docling-PDF_extraction-052FAD)
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
### Techniques Implemented

Apart form standard RAG techniques, searching for improvements that could lead to better answers to cross domain open questions, these techniques were tested:

| Technique | Paper | What It Does |
|-----------|-------|--------------|
| **HyDE** | [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) | Generates hypothetical answers for semantic matching |
| **Query Decomposition** | [arXiv:2507.00355](https://arxiv.org/abs/2507.00355) | Breaks complex questions into sub-queries with RRF merging |
| **Contextual Chunking** | [Anthropic Blog](https://www.anthropic.com/news/contextual-retrieval) | LLM-generated context prepended to chunks (-35% retrieval failures) |
| **RAPTOR** | [arXiv:2401.18059](https://arxiv.org/abs/2401.18059) | Hierarchical summarization tree with UMAP + GMM clustering |
| **GraphRAG** | [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) | Knowledge graph + Leiden communities for cross-document reasoning |
| **GraphRAG Auto-Tuning** | [MS Research](https://www.microsoft.com/en-us/research/blog/graphrag-auto-tuning-provides-rapid-adaptation-to-new-domains/) | Discovers entity types from corpus content (per-book resumable) |

---



### Workflow

The data workflow starts with  books in PDF and follows the standard RAG pipeline. It allows to test different RAG improvement techniques both alone and combined to see the effect of each possible combination chunking strategy/query preprocessing and search type.


![RAGlab workflow](assets/workflow.png)

---
### Documentation

For implementation details, design decisions, and code walkthroughs:

- **[Getting Started](docs/getting-started.md)** — Installation, prerequisites, commands
- **[Architecture](docs/architecture.md)** — Pipeline diagram, project structure
- **[Content Preparation](docs/content-preparation/)** — PDF extraction, cleaning
- **[Chunking Strategies](docs/chunking/)** — Section, Contextual, RAPTOR
- **[Preprocessing Strategies](docs/preprocessing/)** — HyDE, Decomposition, GraphRAG
- **[Evaluation Framework](docs/evaluation/)** — RAGAS metrics and results
---

### Corpus

List of books used to get an idea of the scope and amount of tokens to search over. It is just a set of books of mine selected more or less arbitrarily containing some of the content I want to ask about and relate to each other. The tokens are calculated with *tiktoken* with *cl100k_base encoding* (GPT-4 / *text-embedding-3-large* tokenizer).

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

The evaluation of RAG system can be done in two ways:

- **Streamlit UI**. It allows to change the configuration and see the intermediate and final results. From the UI you can choose:

    - embedding collection from the ones in Weaviate,
    - search type (keyword, hybrid with specific alpha or pure semantic) and top k,
    - reranking used or not, and
    - preprocessing technique applied (HyDE, Query Decomposition, GraphRAG).

    You can write a question and see all the pipeline inermediate results, the chunks retrieved, the score of each chunk, the intermediante LLM interactions (for Query Decomposition or HyDE) and the final answer, so in one place you can easily compare intermediate steps and final results of each configuration for same question.

    ![Streamlit UI](assets/streamlit_ui.png)

- **Running python code**. In addition to user direct evaluation at UI, an evaluation stage is included using RAGAS metrics over a set of handcrafted questions combining single concept and cross domain questions. 


Sample question & answer:

- Question: why do humans care so much about others opinion?
- Answer: 
    Humans care deeply about others' opinions for several intertwined biological, psychological, and social reasons, as highlighted in the provided sources.

    *Social Status and Hierarchies*: Humans are inherently social animals, and our brains are finely tuned to navigate complex social hierarchies. We monitor status differences among individuals, which is crucial for our survival and social cohesion. The amygdala, a brain region associated with emotional responses, becomes particularly active when status relations are ambiguous or shifting, indicating that our emotional well-being is closely tied to how we perceive our standing in social contexts [1]. This suggests that we are not only aware of our own status but are also acutely sensitive to how others perceive us, as these perceptions can influence our social standing and, consequently, our mental health.

    *Theory of Mind and Reputation*: The ability to understand others' thoughts and feelings—known as Theory of Mind—is essential for social interactions. This cognitive skill allows us to gauge how we are viewed by others, which is critical for maintaining our social reputation [6]. Studies have shown that reflecting on one's reputation activates specific brain regions, such as the medial prefrontal cortex, which is involved in self-referential processing and social cognition [6]. Thus, our concern for others' opinions is not merely a social nicety; it is a fundamental aspect of how we navigate our social world.

    *Emotional and Motivational Drivers*: Our emotional responses to social feedback are powerful motivators. For instance, social criticism can activate the amygdala, particularly in individuals with social anxiety, highlighting the distress that negative evaluations can cause [6]. Conversely, positive social feedback can enhance feelings of reward and motivation, reinforcing our desire to be viewed favorably by others [6]. This interplay between social feedback and emotional response underscores why we are so invested in how others perceive us.

    *Cultural and Evolutionary Factors*: From an evolutionary perspective, being attuned to others' opinions has conferred survival advantages. Humans have historically relied on social bonds for cooperation, resource sharing, and protection against threats. Our ability to form complex social structures and maintain relationships is a hallmark of our species, and caring about others' opinions facilitates these connections [10]. This social nature has likely shaped our cognitive processes, making us more sensitive to social cues and evaluations.

    *Philosophical Insights*: Philosophically, thinkers like Epictetus emphasize the importance of understanding our roles and duties within the social fabric. He suggests that our moral intuitions and connections to others guide our actions, indicating that our concern for others' opinions is not merely about social standing but also about fulfilling our responsibilities to those around us [4][9]. This perspective encourages a balance between caring for others' views and maintaining our moral integrity.

    In summary, humans care about others' opinions due to a complex interplay of biological imperatives, cognitive processes, emotional responses, and philosophical understandings of our social roles. This multifaceted concern is deeply embedded in our nature as social beings, shaping our interactions and influencing our mental health and well-being.

    References

    *[1] Behave, The_Biology of Humans at Our Best Worst — Robert M. Sapolsky, Section: SUMMARY AND CONCLUSIONS*

    *[4] The Art of Living ,The Classical Manual on Virtue Happiness and Effectiveness — Epictetus, Section: Our Duties Are Revealed by Our Relations with-One Another*

    *[6] Brain and behavior, a cognitive neuroscience perspective — David Eagleman, Jonathan Downar, Section: Social Reward and Social Aversion*

    *[9] The Art of Living ,The Classical Manual on Virtue Happiness and Effectiveness — Epictetus, Section: Trust Your Moral Intuitions*

    *[10] Fundamentals of Cognitive Neuroscience, A_Beginners Guide — Nicole M. Gage Bernard, Section: Our Social Nature*

    *config: RAPTOR (Hierarchical) | hybrid (α=0.5) | hyde | 10 results | 28,607ms  | Model: openai/gpt-4o-mini*



## Key Insights

Building this pipeline taught me that RAG is deceptively complex:

**PDF parsing is harder than expected.** Scientific books with complex layouts, figures, and footnotes break naive extraction. Docling helped, but significant cleaning was still needed.

**Prompts make or break LLM-based techniques.** HyDE, RAPTOR summarization, and entity extraction all depend heavily on prompt engineering. Small wording changes dramatically affect output quality.

**Evaluation is the hardest part.** Generating good test questions for RAGAS requires domain expertise. The gap between "looks reasonable" and "measurably good" is where real learning happens.

**GraphRAG complexity is justified.** The knowledge graph + Leiden communities approach seemed over-engineered at first, but it handles cross-document reasoning that vector search alone cannot.

