# Step-Back Prompting Research & Improvement Plan

**Created**: December 21, 2024
**Status**: Ready for implementation
**Related**: Phase 1 Quick Wins in `rag-improvement-plan.md`

---

## Executive Summary

The current step-back implementation follows the basic pattern from Google DeepMind's paper but misses several key optimizations. The main issues are:
1. **Single abstraction** - Only one step-back query, losing specificity
2. **No principle extraction** - Missing the "first principles" step from the paper
3. **Poor domain balancing** - The prompt creates generic dual-domain queries that don't target specific concepts
4. **No multi-query retrieval** - Single query can't capture multi-faceted questions

**Implementation strategy**: Start simple with prompt improvements only (Phase 1), measure impact, then iterate with multi-query (Phase 2).

---

## Part 1: Understanding Step-Back Prompting (Theory)

### The Original Technique (arXiv:2310.06117)

Google DeepMind's "Take a Step Back" paper introduced a **two-stage abstraction-then-reasoning process**:

```
Stage 1: ABSTRACTION
"What are the underlying principles or concepts relevant to this question?"

Stage 2: REASONING
Apply those principles to solve the original problem
```

**Key insight**: The paper showed **+27% improvement on TimeQA** and **+7% on multi-hop reasoning** by first extracting abstract principles before answering.

### Why It Works for RAG

In RAG systems, step-back prompting serves a different purpose than in pure reasoning:

| Pure Reasoning | RAG Retrieval |
|----------------|---------------|
| Extract principles to reason better | Extract concepts to retrieve better |
| Answer directly after abstraction | Use abstraction as search query |
| Single-turn process | Query transformation |

**The problem**: The current implementation conflates these - it generates a *single broad query* that loses the original question's specificity.

### Current Implementation vs. Optimal

**Current** (`query_classifier.py:268-286`):
```
Input:  "Why do we need approval from others to feel good?"
Output: "neuroscience of social validation; philosophical ethics of external approval"
```

**Problem**: This query is:
- Too abstract (loses "feel good" → dopamine/reward systems)
- Too narrow (misses evolutionary psychology, belongingness needs)
- Generic structure ("neuroscience of X; philosophical Y")

**Optimal approach** (from research):
```
Step 1 - Extract Principles:
  - Social reward processing in the brain
  - Evolutionary basis of social bonding
  - Philosophical perspectives on external vs internal validation
  - The role of dopamine in social feedback

Step 2 - Generate Multiple Targeted Queries:
  - "dopamine reward social approval brain"
  - "evolutionary psychology social belonging needs"
  - "stoic philosophy external validation virtue"
  - "intrinsic vs extrinsic motivation neuroscience"
```

---

## Part 2: Current Implementation Analysis

### File: `src/preprocessing/query_classifier.py`

**Current STEP_BACK_PROMPT (lines 268-286)**:
```python
STEP_BACK_PROMPT = """You help improve search queries for a knowledge system about human behavior.

The system contains neuroscience (brain, behavior, consciousness) AND philosophy (wisdom, ethics, meaning).

Given the user's question, generate a broader "step-back" query that will retrieve relevant content
from both scientific AND philosophical perspectives when appropriate.

Principles:
- Identify the UNDERLYING TOPIC (emotion, motivation, decision-making, meaning, consciousness, etc.)
- Include SCIENTIFIC ANGLE (brain mechanisms, neurotransmitters, psychology, evolution)
- Include WISDOM ANGLE (philosophical practices, ancient wisdom, life guidance)
- Keep it broad enough to catch diverse relevant passages

Examples of the pattern:
- Specific question -> "underlying topic from neuroscience; underlying topic from philosophy"
- "Why do I feel anxious?" -> "neuroscience of anxiety and fear; philosophical approaches to tranquility"
- "What is the point of life?" -> "psychology of meaning and purpose; philosophical perspectives on the good life"

Generate ONLY the step-back query. Keep it under 20 words."""
```

### Problems Identified

1. **Forced dual-domain structure**: The prompt forces "X from neuroscience; Y from philosophy" pattern, which:
   - Creates generic queries that don't match actual chunk content
   - Misses domain-specific terminology that would improve retrieval
   - Assumes every question needs both domains equally

2. **Single query limitation**: Only one search query is generated, but multi-faceted questions need multi-query retrieval

3. **No principle extraction**: The prompt skips the crucial "what are the underlying concepts" step

4. **Too abstract**: "neuroscience of social validation" is less retrievable than "dopamine reward social feedback anterior cingulate"

5. **No domain-specific vocabulary**: Chunks contain specific terms (authors, brain regions, philosophical schools) that this prompt doesn't elicit

---

## Part 3: Advanced Techniques from Research

### Technique 1: Principle Extraction First (from original paper)

**Add an explicit principle-extraction step before query generation**:

```python
PRINCIPLE_EXTRACTION_PROMPT = """Identify the key concepts and principles underlying this question.

Question: {query}

List 3-5 core concepts that someone would need to understand to fully answer this:
1. [Concept from psychology/neuroscience]
2. [Concept from philosophy/wisdom]
3. [Related mechanisms or processes]
4. [Key terms or vocabulary]
5. [Related authors/traditions if applicable]

Format: JSON array of concepts
"""
```

### Technique 2: Multi-Query Generation (from Query Decomposition research)

**Generate multiple targeted queries instead of one broad query**:

```python
MULTI_QUERY_PROMPT = """Generate 3-4 targeted search queries to find relevant information.

Original question: {query}
Underlying concepts: {principles}

Generate queries that:
- Use domain-specific vocabulary (brain regions, neurotransmitters, philosopher names)
- Target different aspects of the question
- Include both specific terms AND conceptual phrases

Format: JSON array of queries
"""
```

### Technique 3: Reciprocal Rank Fusion (RRF)

When using multiple queries, merge results using RRF:

```python
def reciprocal_rank_fusion(results_lists: List[List], k: int = 60) -> List:
    """Merge multiple result lists using RRF scoring."""
    scores = defaultdict(float)
    for results in results_lists:
        for rank, item in enumerate(results):
            scores[item.id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

---

## Part 4: Recommended Improved Prompts

### For Future Multi-Query Implementation

#### PRINCIPLE_EXTRACTION_PROMPT

```python
PRINCIPLE_EXTRACTION_PROMPT = """You are analyzing a question about human nature for a knowledge retrieval system.

The knowledge base contains:
- NEUROSCIENCE: Brain mechanisms, neurotransmitters (dopamine, serotonin, oxytocin), brain regions (prefrontal cortex, amygdala, insula), emotions, decision-making, consciousness
- PHILOSOPHY: Stoicism (Marcus Aurelius, Epictetus, Seneca), Taoism (Lao Tzu, Chuang Tzu), Buddhism, Existentialism, virtue ethics, meaning, wisdom traditions

Given this question, extract the KEY UNDERLYING CONCEPTS that would help retrieve relevant passages:

Question: "{query}"

Identify:
1. CORE TOPIC: What is the fundamental subject? (e.g., "social reward", "anxiety", "purpose")
2. NEUROSCIENCE CONCEPTS: Specific mechanisms, brain regions, or processes (e.g., "dopamine reward pathway", "amygdala fear response")
3. PHILOSOPHICAL CONCEPTS: Relevant schools, authors, or ideas (e.g., "Stoic indifference to externals", "Taoist non-attachment")
4. RELATED TERMS: Vocabulary likely to appear in relevant passages

Respond with JSON:
{
  "core_topic": "...",
  "neuroscience_concepts": ["...", "..."],
  "philosophical_concepts": ["...", "..."],
  "related_terms": ["...", "..."]
}
"""
```

#### MULTI_QUERY_PROMPT

```python
MULTI_QUERY_PROMPT = """Generate targeted search queries for a hybrid neuroscience + philosophy knowledge base.

Original question: "{query}"

Extracted concepts:
- Core topic: {core_topic}
- Neuroscience: {neuro_concepts}
- Philosophy: {philo_concepts}
- Related terms: {related_terms}

Generate 4 search queries that will retrieve diverse, relevant passages:

1. A NEUROSCIENCE-focused query using specific brain/psychological terms
2. A PHILOSOPHY-focused query using specific traditions/authors
3. A BRIDGING query that connects scientific and philosophical perspectives
4. A BROAD query using the core topic in accessible language

Each query should be 5-12 words. Use vocabulary from the knowledge base.

Respond with JSON:
{
  "queries": [
    {"type": "neuroscience", "query": "..."},
    {"type": "philosophy", "query": "..."},
    {"type": "bridging", "query": "..."},
    {"type": "broad", "query": "..."}
  ]
}
"""
```

### Example Transformation

**Input**: "Why do we need approval from others to feel good?"

**Step 1 - Principle Extraction**:
```json
{
  "core_topic": "social validation and emotional reward",
  "neuroscience_concepts": [
    "dopamine reward system",
    "social reward processing",
    "ventral striatum activation",
    "oxytocin social bonding"
  ],
  "philosophical_concepts": [
    "Stoic indifference to external judgment",
    "intrinsic vs extrinsic sources of worth",
    "virtue as sole good",
    "Taoist naturalness"
  ],
  "related_terms": [
    "self-esteem", "belonging", "rejection", "praise",
    "Marcus Aurelius", "Epictetus", "social brain"
  ]
}
```

**Step 2 - Multi-Query Generation**:
```json
{
  "queries": [
    {"type": "neuroscience", "query": "dopamine reward social approval ventral striatum brain"},
    {"type": "philosophy", "query": "Stoic indifference external judgment virtue Epictetus"},
    {"type": "bridging", "query": "social validation psychological need philosophical response"},
    {"type": "broad", "query": "why humans seek approval belonging self-worth"}
  ]
}
```

**Result**: 4 queries retrieve diverse chunks, then RRF merges them with diversity balancing.

---

## Part 5: Implementation Plan (Iterative Approach)

**Strategy**: Start simple with prompt improvements only, measure impact, then iterate.

### Phase 1: Improve Step-Back Prompt (Current Focus)

**File**: `src/preprocessing/query_classifier.py`

**Change**: Replace `STEP_BACK_PROMPT` with an improved version that:
1. Explicitly extracts principles first (in the same prompt via Chain-of-Thought examples)
2. Uses domain-specific vocabulary from the book collection
3. Generates a richer, more targeted query

#### Improved Prompt:

```python
STEP_BACK_PROMPT = """You transform questions into effective search queries for a knowledge system about human nature.

KNOWLEDGE BASE CONTENTS:
- Neuroscience books: brain mechanisms, dopamine/serotonin/oxytocin, prefrontal cortex, amygdala, limbic system, decision-making, emotions, consciousness, evolutionary psychology
- Philosophy books: Stoicism (Marcus Aurelius, Epictetus, Seneca), Taoism (Lao Tzu, Chuang Tzu), Buddhism, virtue ethics, meaning of life, wisdom traditions

TASK: Generate a search query that will retrieve the most relevant passages.

PROCESS:
1. Identify the CORE TOPIC: What is the user really asking about? (e.g., fear, purpose, social needs, self-control)
2. Identify SPECIFIC MECHANISMS: What brain systems, psychological processes, or philosophical concepts relate?
3. Use CONCRETE VOCABULARY: Include specific terms from the knowledge base (author names, brain regions, philosophical schools, emotions)

EXAMPLES:
User: "Why do I feel anxious?"
Think: Core=anxiety/fear, Mechanisms=amygdala+cortisol+fight-or-flight+Stoic tranquility
Query: "amygdala fear response anxiety Stoic tranquility ataraxia Epictetus control"

User: "Why do we need approval from others to feel good?"
Think: Core=social validation+reward, Mechanisms=dopamine+social brain+oxytocin+Stoic indifference to externals
Query: "dopamine social reward approval seeking Stoic virtue external validation Marcus Aurelius"

User: "What is the point of life?"
Think: Core=meaning/purpose, Mechanisms=prefrontal cortex goal-setting+existential psychology+Stoic eudaimonia+Taoist wu-wei
Query: "meaning purpose life eudaimonia Stoicism Taoism Viktor Frankl prefrontal goals"

User: "How can I control my anger?"
Think: Core=anger regulation, Mechanisms=amygdala+prefrontal inhibition+Seneca on anger+cognitive reappraisal
Query: "anger regulation amygdala prefrontal Seneca De Ira Stoic passion cognitive reappraisal"

Generate ONLY the search query. Use 10-20 words. Include both neuroscience and philosophy terms."""
```

### Key Differences from Current Prompt

| Aspect | Current | Improved |
|--------|---------|----------|
| Structure | Generic "X from neuro; Y from philo" | Concrete terms mixed together |
| Vocabulary | Abstract ("brain mechanisms") | Specific ("amygdala", "dopamine", "Epictetus") |
| Process | Direct generation | Think-then-generate (Chain-of-Thought in prompt) |
| Examples | 2 simple examples | 4 detailed examples with reasoning traces |
| Length | Under 20 words | 10-20 words (allows more specificity) |

### Why This Works Better

1. **Domain vocabulary**: Uses actual terms from the books (Marcus Aurelius, amygdala, eudaimonia) that match chunk content

2. **Chain-of-Thought in examples**: Shows the model HOW to think ("Think: Core=..., Mechanisms=...") before generating

3. **Mixed domain terms**: Instead of separate neuroscience/philosophy sections, terms are intermixed for better hybrid retrieval

4. **Concrete examples**: 4 examples covering different question types (anxiety, social, meaning, anger)

### No Code Logic Changes Required

- Same `step_back_prompt()` function
- Same return format (single string)
- Same API call parameters
- Only the prompt text changes

### Evaluation Plan

After implementing, run:
```bash
python -m src.run_stage_7_evaluation --collection [your_collection]
```

Compare:
- Context precision (are retrieved chunks relevant?)
- Context recall (did we find all relevant chunks?)
- Diversity balance (60/40 neuro/philo split maintained?)

### Phase 2: Multi-Query Generation (Future)

If Phase 1 prompt improvement shows gains:
1. Add `extract_principles()` function
2. Add `generate_multi_queries()` function
3. Implement RRF merging in retrieval
4. Update `PreprocessedQuery` dataclass with new fields

### Phase 3: MULTI_HOP Query Decomposition (Future)

Handle explicit comparison queries like "Compare Stoic and Buddhist views on suffering":
1. Add `decompose_query()` function
2. Generate sub-queries for each aspect
3. Merge results with RRF

---

## Part 6: Evaluation Metrics

Track these before/after:
1. **Retrieval diversity**: % neuro vs % philo in top-10
2. **Relevance scores**: Average reranker score
3. **RAGAS metrics**: Context precision, context recall
4. **Query coverage**: Do sub-queries find unique chunks?

---

## Sources

- [Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models](https://arxiv.org/abs/2310.06117) - Original step-back paper (Google DeepMind, 2023)
- [Query Decomposition for RAG](https://arxiv.org/html/2507.00355v1) - Multi-query retrieval with RRF
- [Step-Back Prompting - Learn Prompting](https://learnprompting.org/docs/advanced/thought_generation/step_back_prompting) - Implementation guide
- [Advanced RAG: Query Decomposition - Haystack](https://haystack.deepset.ai/blog/query-decomposition) - Practical patterns
- [In-Depth RAG Query Transformation](https://dev.to/jamesli/in-depth-understanding-of-rag-query-transformation-optimization-multi-query-problem-decomposition-and-step-back-27jg) - Comparison of techniques
