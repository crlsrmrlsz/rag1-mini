"""Retrieval module for advanced search operations.

## RAG Theory: Multi-Query Retrieval

Single queries often miss relevant documents due to vocabulary mismatch
or conceptual gaps. Multi-query retrieval addresses this by:

1. Generating multiple query variants (different phrasings, concepts)
2. Retrieving for each query independently
3. Merging results using Reciprocal Rank Fusion (RRF)

RRF is preferred over score-based merging because:
- No score normalization needed across different queries
- Robust to outlier scores
- Simple and effective (standard in RAG research)

## Usage

```python
from src.retrieval import reciprocal_rank_fusion, RRFResult

# After retrieving for multiple queries
rrf_result = reciprocal_rank_fusion(
    result_lists=[results_q1, results_q2, results_q3],
    query_types=["original", "neuroscience", "philosophy"],
    top_k=10
)

# Access merged results
for result in rrf_result.results:
    print(f"{result.chunk_id}: RRF score = {result.score:.4f}")
```
"""

from src.retrieval.rrf import reciprocal_rank_fusion, RRFResult, RRF_K


__all__ = [
    "reciprocal_rank_fusion",
    "RRFResult",
    "RRF_K",
]
