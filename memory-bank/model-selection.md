# OpenRouter Model Selection for RAGAS Evaluation

**Research Date:** December 19, 2025
**Selected Tier:** Quality (~$0.50 for 10 questions)

---


## Latest Model Releases (Nov-Dec 2025)

| Date | Model | Highlights |
|------|-------|------------|
| Dec 17 | **Gemini 3 Flash** | New default, 3x faster than 2.5 Pro |
| Dec 11 | **GPT-5.2** | 400K context, 3 tiers (Instant/Thinking/Pro) |
| Nov 24 | **Claude Opus 4.5** | 80.9% SWE-bench, best coding model |
| Nov 18 | **Gemini 3 Pro** | #1 on GPQA (91.9%), leads benchmarks |

---

## Pricing Summary (December 2025)

| Model | OpenRouter ID | Input/1M | Output/1M | Context |
|-------|---------------|----------|-----------|---------|
| **GPT-5 Nano** | `openai/gpt-5-nano` | $0.05 | $0.40 | 128K |
| **GPT-5 Mini** | `openai/gpt-5-mini` | $0.25 | $2.00 | 200K |
| **DeepSeek V3.2** | `deepseek/deepseek-chat` | $0.28 | $0.42 | 128K |
| **Gemini 3 Flash** | `google/gemini-3-flash` | $0.50 | $3.00 | 1M |
| **Claude Haiku 4.5** | `anthropic/claude-haiku-4.5` | $1.00 | $5.00 | 200K |
| **GPT-5.2 Thinking** | `openai/gpt-5.2` | $1.75 | $14.00 | 400K |
| **Gemini 3 Pro** | `google/gemini-3-pro` | $2.00 | $12.00 | 200K |
| **Claude Sonnet 4.5** | `anthropic/claude-sonnet-4.5` | $3.00 | $15.00 | 200K |
| **Claude Opus 4.5** | `anthropic/claude-opus-4.5` | $5.00 | $25.00 | 200K |
| **GPT-5.2 Pro** | `openai/gpt-5.2-pro` | $21.00 | $168.00 | 400K |

---

## Benchmark Performance (December 2025)

| Model | SWE-bench | GPQA Diamond | ARC-AGI-2 | Notes |
|-------|-----------|--------------|-----------|-------|
| **Claude Opus 4.5** | **80.9%** | 83.4% | 37.6% | Best coding |
| **Gemini 3 Pro** | - | **91.9%** | 45.1% | Best reasoning |
| **GPT-5.2 Pro** | - | - | **54.2%** | Best ARC-AGI |
| **Claude Sonnet 4.5** | 77.2% | - | - | Great value |

---

## Key Insights

1. **For Generation:** Gemini 3 Flash offers PhD-level reasoning at $0.50/$3.00 per 1M tokens
2. **For Evaluation (LLM-as-Judge):** RAGAS research shows Anthropic models are most stable
3. **Best Value:** DeepSeek V3.2 at $0.28/$0.42 is extremely cost-effective
4. **Context Windows:** GPT-5.2 leads with 400K, Gemini 3 Flash has 1M

---

## Generation Models for RAG Answers

Models used in the UI for synthesizing answers from retrieved chunks.

| Model | OpenRouter ID | Input/1M | Output/1M | Best For |
|-------|---------------|----------|-----------|----------|
| **GPT-5 Nano** | `openai/gpt-5-nano` | $0.05 | $0.40 | Budget/testing |
| **DeepSeek V3.2** | `deepseek/deepseek-chat` | $0.28 | $0.42 | Best value |
| **GPT-5 Mini** | `openai/gpt-5-mini` | $0.25 | $2.00 | Balanced (default) |
| **Gemini 3 Flash** | `google/gemini-3-flash` | $0.50 | $3.00 | Quality |
| **Claude Haiku 4.5** | `anthropic/claude-haiku-4.5` | $1.00 | $5.00 | Premium |

### Cost per Query (Estimated)

Assuming ~3K input tokens (context + prompt) and ~500 output tokens per answer:

| Model | Cost/Query | 100 Queries |
|-------|-----------|-------------|
| GPT-5 Nano | ~$0.0004 | ~$0.04 |
| DeepSeek V3.2 | ~$0.001 | ~$0.10 |
| GPT-5 Mini | ~$0.002 | ~$0.18 |
| Gemini 3 Flash | ~$0.003 | ~$0.30 |
| Claude Haiku 4.5 | ~$0.006 | ~$0.55 |

### Recommendations

1. **Development/Testing:** GPT-5 Nano - negligible cost
2. **Daily Use:** GPT-5 Mini - good quality at low cost
3. **Quality Focus:** Gemini 3 Flash - excellent reasoning
4. **Production:** Claude Haiku 4.5 - reliable, well-calibrated

---

## Sources

- [OpenRouter Models](https://openrouter.ai/models)
- [Claude Opus 4.5 Announcement](https://www.anthropic.com/news/claude-opus-4-5)
- [GPT-5.2 Introduction](https://openai.com/index/introducing-gpt-5-2/)
- [Gemini 3 Flash Launch](https://blog.google/products/gemini/gemini-3-flash/)
- [RAGAS: Evaluating the Evaluators](https://blog.ragas.io/evaluating-the-evaluators)
