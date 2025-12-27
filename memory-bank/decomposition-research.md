# Query Decomposition Research

**Date:** December 27, 2024
**Purpose:** Research findings for aligning query decomposition with paper and best practices

---

## 1. Foundational Paper

**Title:** "Question Decomposition for Retrieval-Augmented Generation"
**Authors:** Paul J. L. Ammann, Jonas Golde, Alan Akbik (Humboldt-Universität zu Berlin)
**ArXiv:** [2507.00355](https://arxiv.org/abs/2507.00355) (July 2025)
**Results:** +36.7% MRR@10, +11.6% F1 on MultiHop-RAG and HotpotQA

### Core Methodology
1. LLM decomposes original query into sub-questions
2. Passages retrieved for each sub-question
3. Original query always retained alongside sub-queries
4. Cross-encoder reranker scores passages against original query
5. Merged candidate pool returned

### Paper's Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Temperature** | 0.8 | Higher creativity for diverse decompositions |
| **Sampling** | Nucleus (Top-p = 0.8) | Diverse generation |
| **Max sub-questions** | 5 | Enforced limit in experiments |
| **Actual generation** | Exactly 5 (93-99% of time) | LLM applies fixed "budget" |
| **Original query** | Always retained | Critical for context preservation |

### Paper's Prompt
The paper does not disclose the exact prompt wording. It describes using "a fixed natural language prompt provided to an instruction-tuned language model" with zero-shot prompting.

---

## 2. Community Best Practices

### Haystack (Deepset)
```
"Your job is to simplify complex queries into multiple queries that can be answered
in isolation to each other. If the query is simple, then keep it as it is."
```
- Key feature: **"keep it as is" for simple queries**
- Source: [Haystack Blog](https://haystack.deepset.ai/blog/query-decomposition)

### LangChain
```
"Perform query decomposition. Given a user question, break it down into distinct
sub questions that you need to answer in order to answer the original question.
If there are acronyms or words you are not familiar with, do not try to rephrase them."
```
- Key feature: **"do not try to rephrase"** - preserve terminology
- Source: [LangChain Docs](https://python.langchain.com/v0.1/docs/use_cases/query_analysis/techniques/decomposition/)

### EfficientRAG (EMNLP 2024)
```
"You should decompose the given multi-hop question into multiple single-hop
questions, and such that you can answer each single-hop question independently."
```
- Key feature: **"independently"** for parallel retrieval
- Source: [arXiv:2408.04259](https://arxiv.org/abs/2408.04259)

---

## 3. Implementation: Paper-Aligned Prompt (Updated Dec 2024)

### Current Decomposition Prompt
```python
DECOMPOSITION_PROMPT = """Break down this question for a knowledge base on cognitive science and philosophy.

If the question is simple enough to answer directly, keep it as a single question.
Otherwise, create 3-5 sub-questions that can be answered independently and together cover all aspects of the original.

Question: {query}

Respond with JSON:
{{
  "sub_questions": ["...", "...", "..."],
  "reasoning": "Brief explanation"
}}"""
```

### Design Rationale

| Best Practice | Source | Implemented |
|---------------|--------|-------------|
| "If simple, keep as is" | Haystack | Yes |
| "answered independently" | Haystack/EfficientRAG | Yes |
| "cover all aspects" | LangChain | Yes |
| "3-5 sub-questions" | Paper (max 5) | Yes |
| Domain framing | Consistent with HyDE | Yes |

### Parameters
- **Temperature**: 0.7 (paper uses 0.8; 0.7 balances diversity with coherence)
- **Max tokens**: 400
- **Original query**: Always retained in RRF merge

---

## 4. Key Improvements from Alignment

| Aspect | Before | After | Benefit |
|--------|--------|-------|---------|
| Simple queries | Always decomposed | "Keep as single question" | Avoids unnecessary decomposition |
| Independence | Not emphasized | Explicit requirement | Enables parallel retrieval |
| Comprehensiveness | "target specific aspect" | "cover all aspects" | Improves recall |
| Temperature | 0.3 | 0.7 | More diverse sub-questions |
| Sub-question count | 3-4 | 3-5 | Closer to paper's max 5 |

---

## 5. Sources

### Primary Research
- [arXiv:2507.00355 - Question Decomposition for RAG](https://arxiv.org/abs/2507.00355)
- [EfficientRAG Paper (arXiv:2408.04259)](https://arxiv.org/abs/2408.04259)

### Implementation Guides
- [Haystack Query Decomposition Blog](https://haystack.deepset.ai/blog/query-decomposition)
- [Haystack Query Decomposition Cookbook](https://haystack.deepset.ai/cookbook/query_decomposition)
- [LangChain Decomposition Docs](https://python.langchain.com/v0.1/docs/use_cases/query_analysis/techniques/decomposition/)
- [LlamaIndex Query Transform Cookbook](https://developers.llamaindex.ai/python/examples/query_transformations/query_transform_cookbook/)
- [RAG Techniques Repository](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/query_transformations.ipynb)
