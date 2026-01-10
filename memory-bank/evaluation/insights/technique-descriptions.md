# Technique Descriptions for Evaluation Interpretation

This document provides one-paragraph descriptions of each technique used in comprehensive evaluation. Each description captures the essential transformation the technique applies to data and its effects on RAG performance, enabling interpretation of evaluation statistics.

---

## Search Types

### Keyword (BM25)

Performs pure lexical matching using term frequency and inverse document frequency scoring, finding chunks that contain the exact words in the query without any semantic understanding. This approach excels when queries use precise technical terminology (like "amygdala" or "thalamocortical") but completely fails for synonyms, paraphrases, or conceptual descriptions that use different vocabulary. In RAGLab evaluation, keyword search achieves best single-concept answer correctness (56.8%) but worst cross-domain performance (-11.49% degradation), suggesting it works well when query terms match document terminology exactly but struggles with abstract philosophical concepts that may be described in diverse ways across different authors.

### Hybrid (with Alpha)

Combines vector similarity (semantic understanding) and BM25 (keyword matching) using a weighted parameter alpha (0.0 = pure keyword, 1.0 = pure vector, 0.5 = balanced). This dual-signal approach captures both conceptual meaning AND exact terminology, making it robust to vocabulary variation while still matching specific terms. The alpha parameter allows tuning: higher values (0.6-0.8) favor semantic matching for abstract concepts, lower values (0.3-0.5) favor keyword matching for technical terminology. In evaluation, hybrid achieves best single-concept recall (97.1%) but moderate cross-domain performance, as the BM25 component helps precision while vectors help recall—the blend provides consistent baseline performance across query types without excelling at either extreme.

---

## Chunking Strategies

### Section Chunking

Splits documents into sequential 800-token chunks with 2-sentence overlap at boundaries, respecting document structure (chapters, sections) by never crossing section boundaries. This preserves the author's intended reading order and narrative flow—a philosophical argument builds across sentences in natural progression. The overlap ensures boundary concepts aren't orphaned, and section boundaries prevent mixing unrelated topics. In evaluation, section chunking provides the most stable baseline: best single-concept relevancy (89.1%) and smallest consistency degradation across query types (-16.6%), suggesting it works reliably when documents have clear structural organization and queries target content within a single section's scope.

### Contextual Chunking

Enhances section chunks by prepending a 2-3 sentence LLM-generated summary that situates each chunk within its broader document context. A chunk containing "The author argues against reductionist accounts" becomes "[This passage from Chapter 5 of Chalmers' work critiques physicalist theories of consciousness...] The author argues against reductionist accounts." This disambiguates pronouns, resolves references, and establishes thematic scope before embedding. In evaluation, contextual achieves best answer correctness for both single-concept (59.1%) and cross-domain (48.8%) queries, with exceptional recall (96.3%)—the added context helps the embedding model understand what each chunk is "about" rather than just what words it contains, enabling retrieval of chunks that are conceptually relevant even when terminology differs.

### RAPTOR (Hierarchical Summarization Tree)

Builds a multi-level tree on top of section chunks by embedding all chunks, clustering semantically similar ones (using UMAP dimensionality reduction + Gaussian Mixture Models), generating LLM summaries of each cluster, then recursively repeating on summaries to create 3-4 abstraction levels. Discussions of "consciousness" scattered across 100 pages cluster together despite distance in the document. The vector store contains both leaf chunks (for specific details) and summary nodes (for thematic overview), with retrieval automatically selecting the appropriate granularity based on query similarity. In evaluation, RAPTOR achieves best faithfulness (95.2%) and best cross-domain precision (93.8%), suggesting the hierarchical summaries reduce hallucination by providing verified thematic anchors and work particularly well for cross-domain queries that need high-level conceptual understanding rather than specific passages.

---

## Preprocessing Strategies

### None (Baseline)

Passes the user's query directly to search without any transformation, relying entirely on the embedding model to bridge the gap between question language and document language. This zero-overhead approach works surprisingly well for straightforward queries where the embedding model's training covers the semantic space, avoiding LLM latency and potential hallucination in transformation. In evaluation, "none" achieves competitive single-concept answer correctness (56.3%) and relevancy (88.4%), suggesting that for simple queries targeting familiar concepts, the embedding model already understands the semantic relationship—adding preprocessing can introduce noise without proportional benefit.

### HyDE (Hypothetical Document Embeddings)

Generates 2-5 hypothetical answer passages using an LLM before search, then embeds these hypotheticals (averaging their embeddings) to search for similar actual documents. The insight is that two answers to the same question are semantically closer than a question and its answer—"What causes stress?" is far from a passage about cortisol, but a hypothetical answer about cortisol is close to the real passage. In evaluation, HyDE achieves best cross-domain recall (78.8%) and smallest cross-domain degradation (-10.5%), suggesting it effectively bridges the semantic gap for complex queries by generating domain-appropriate vocabulary and conceptual framings that match how authors actually write about topics, even when the user's question uses different terminology.

### Decomposition

Breaks complex multi-aspect questions into 2-4 simpler sub-questions using an LLM, searches for each independently, and merges results using Reciprocal Rank Fusion (RRF). A question like "How do Eastern and Western philosophies differ on consciousness?" becomes sub-questions targeting each tradition separately. This ensures each facet gets dedicated retrieval attention rather than being averaged out in a single search. In evaluation, decomposition achieves best single-concept precision (73.8%) and recall (96.0%), but worst cross-domain recall (65.6% with -30.4% degradation)—the strategy excels for multi-step questions within a single domain but fragments retrieval coherence when sub-questions need to be integrated across conceptual boundaries, as RRF merging cannot reconstruct the synthesis the original query implied.

### GraphRAG

Extracts entities and relationships from the query using an LLM, traverses a pre-built knowledge graph (Neo4j with Leiden community detection) to find related concepts, then merges graph-derived chunks with vector search results. For "What is consciousness?", it identifies entities like "consciousness", "phenomenal experience", "neural correlates", traverses their relationships to find connected concepts like "qualia", "hard problem", "Chalmers", and boosts chunks that appear in both graph traversal and vector search. In evaluation, GraphRAG achieves best overall answer correctness for both single-concept (59.5%, +5.7% over baseline) and cross-domain (50.1%, +5.2%)—the knowledge graph structure captures conceptual relationships that pure vector similarity misses, particularly valuable when questions involve how concepts relate to each other rather than just what they mean independently.

---

## Interaction Effects and Combination Nuances

### Synergistic Combinations

**Contextual + GraphRAG**: The LLM-generated context in chunks synergizes with graph structure—context disambiguates "what this chunk is about" while the graph captures "how concepts relate," providing both local clarity and global structure.

**RAPTOR + Hybrid**: Hierarchical summaries contain broader vocabulary that benefits from keyword matching (BM25 component), while specific leaf chunks benefit from semantic matching (vector component)—hybrid search naturally routes to the appropriate abstraction level.

**HyDE + Keyword**: Hypothetical generation compensates for keyword search's vocabulary limitation by generating domain-appropriate terms, allowing BM25 to match against generated terminology rather than user's potentially unfamiliar phrasing (+13% recall improvement).

### Anti-Patterns

**Decomposition + Cross-domain**: Fragmenting cross-domain questions loses the essential synthesis requirement—when sub-questions target different domains, RRF merging cannot reconstruct the conceptual integration the original query needed, leading to retrieved chunks that answer parts but not the whole.

**Semantic 0.75 Chunking**: Loose similarity thresholds create chunks with weak internal coherence, underperforming everywhere in evaluation.

**Keyword + Semantic-chunked Content**: Semantically-chunked content assumes semantic retrieval; keyword matching misses the conceptual boundaries that defined the chunks.

---

## Quick Reference Table

| Technique | Type | Key Transformation | Best For | Worst For |
|-----------|------|-------------------|----------|-----------|
| Keyword | Search | Exact term matching | Technical terminology | Synonyms, paraphrases |
| Hybrid | Search | Vector + BM25 fusion | Balanced retrieval | Neither extreme |
| Section | Chunking | Sequential + overlap | Structured documents | Scattered concepts |
| Contextual | Chunking | LLM context prepended | Answer correctness | Cost-sensitive use |
| RAPTOR | Chunking | Hierarchical tree | Cross-domain precision | Simple queries |
| None | Preprocessing | Pass-through | Simple queries | Complex synthesis |
| HyDE | Preprocessing | Hypothetical generation | Cross-domain recall | Overhead-sensitive |
| Decomposition | Preprocessing | Sub-question split | Multi-step single-domain | Cross-domain synthesis |
| GraphRAG | Preprocessing | Entity graph traversal | Relationship queries | Infrastructure cost |

---

## Usage Notes

These descriptions help interpret evaluation results by understanding:

1. **Why certain combinations work**: Synergies between techniques that address complementary weaknesses
2. **Why certain combinations fail**: Anti-patterns where techniques interfere with each other
3. **What metrics each technique optimizes**: Precision vs recall vs answer correctness trade-offs
4. **Domain sensitivity**: Which techniques work better for technical vs philosophical content

When analyzing evaluation statistics, consider:
- The underlying semantic transformation each technique applies
- Whether the query type matches the technique's strengths
- How multiple techniques in a pipeline interact (sequential effects)
- The cost-benefit trade-off for infrastructure and LLM overhead
