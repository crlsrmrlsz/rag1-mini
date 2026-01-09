# Evaluation Design

This document consolidates the evaluation framework design, corpus analysis, and test question design for RAGLab.

---

## 1. Benchmark Context

### Standard QA Benchmarks (RAPTOR Paper)

The RAPTOR paper ([arXiv:2401.18059](https://arxiv.org/abs/2401.18059)) evaluates on three benchmark datasets:

| Dataset | Domain | Size | Document Length | Answer Type | Primary Metric |
|---------|--------|------|-----------------|-------------|----------------|
| **QASPER** | NLP research papers | 5,049 Q / 1,585 papers | ~3K-5K tokens | Extractive, Yes/No, Free-form | Token F1 |
| **QuALITY** | Fiction stories | ~2,500 Q / 250 stories | ~5K tokens | Multiple-choice (4 options) | Accuracy |
| **NarrativeQA** | Books + movie scripts | 46,765 Q&A pairs | 60K+ tokens (books) | Free-form (synthesized) | F1 + BLEU |

**QASPER**: Questions written from paper abstracts, answered from full text. 55.5% require multi-paragraph evidence. Source: [HuggingFace](https://huggingface.co/datasets/allenai/qasper)

**QuALITY**: Multiple-choice format eliminates answer ambiguity. "Hard" subset: questions that annotators couldn't answer quickly. Source: [NYU-MLL](https://nyu-mll.github.io/quality/)

**NarrativeQA**: Only 29.6% extractive - most answers require synthesis. Two reference answers per question. Source: [GitHub](https://github.com/google-deepmind/narrativeqa)

### Why Domain-Specific Evaluation?

RAGLab uses **domain-specific evaluation** rather than generic benchmarks because:

1. QASPER tests NLP papers (not neuroscience/philosophy)
2. QuALITY tests fiction stories (different document structure)
3. NarrativeQA tests fiction books (not technical content)

Domain-specific questions better measure retrieval quality for the actual corpus.

### Answer Correctness Metric

RAGAS `AnswerCorrectness` is the end-to-end metric:

```python
from ragas.metrics import AnswerCorrectness

# Weighted combination:
# - Factual similarity (75%): LLM decomposes into claims, classifies TP/FP/FN
# - Semantic similarity (25%): Embedding cosine similarity
```

**Pros**: Understands synonyms, paraphrasing, semantic equivalence
**Cons**: Slow (~2-5s/question), costs LLM calls, non-deterministic

**Decision**: SQuAD-style token F1 was removed (Dec 2024) because it lacks semantic understanding.

---

## 2. Corpus Analysis

Analysis of 19 books (6,249 chunks) to identify cross-cutting themes between neuroscience and wisdom traditions.

### Book Inventory

**Neuroscience/Psychology (9 books, ~4,800 chunks)**

| Book | Key Themes | Chunk Count |
|------|-----------|-------------|
| **Behave** (Sapolsky) | Violence, empathy, social behavior, morality | ~400 |
| **Biopsychology** (Pinel/Barnes) | Sleep, memory, addiction, brain structure | ~600 |
| **Brain and Behavior** (Eagleman/Downar) | Consciousness, emotions, motivation, reward | ~950 |
| **Cognitive Biology** (Tommasi et al.) | Evolution of cognition, memory, social cognition | ~130 |
| **Cognitive Neuroscience** (Gazzaniga) | Consciousness, memory, split-brain, emotions | ~880 |
| **Determined** (Sapolsky) | Free will, determinism, moral responsibility | ~270 |
| **Fundamentals of Cog Neuro** (Gage/Bernard) | Consciousness, sleep, emotions, social brain | ~640 |
| **Psychobiology of Behaviour** (Fountoulakis) | Sleep, reward, addiction, aggression | ~500 |
| **Thinking Fast and Slow** (Kahneman) | Decision biases, heuristics, System 1/2 | ~480 |

**Wisdom/Philosophy (10 books, ~1,400 chunks)**

| Book | Key Themes | Chunk Count |
|------|-----------|-------------|
| **Essays and Aphorisms** (Schopenhauer) | Suffering, will, consciousness, death | ~150 |
| **Letters from a Stoic** (Seneca) | Time, death, anger, friendship, adversity | ~200 |
| **Tao te ching** (Lao Tzu) | Non-action, balance, humility, detachment | ~80 |
| **The Analects** (Confucius) | Virtue, benevolence, ritual, learning | ~150 |
| **The Art of Living** (Epictetus) | Control, virtue, impressions, self-mastery | ~100 |
| **The Enchiridion** (Epictetus) | Free will, good/evil, roles, divine will | ~50 |
| **The Meditations** (Marcus Aurelius) | Self-discipline, death, reason, tranquility | ~200 |
| **The Pocket Oracle** (Gracian) | Prudence, reputation, discretion, strategy | ~70 |
| **Essays, counsels, maxims** (Schopenhauer) | Happiness, solitude, relationships | ~150 |
| **Wisdom of Life** (Schopenhauer) | Personality, health, honor, contentment | ~70 |

### Cross-Domain Theme Matrix

| # | Theme | Neuroscience Sources | Philosophy Sources | Coverage |
|---|-------|---------------------|-------------------|----------|
| 1 | **Free will & determinism** | Determined (271 refs), Cognitive Neuro | Stoics (fate), Schopenhauer (will), Taoism (wu wei) | STRONG |
| 2 | **Self-control & willpower** | Kahneman (System 1/2), frontal cortex, Behave | Epictetus, Marcus Aurelius, Seneca | STRONG |
| 3 | **Emotional regulation** | All neuro books (67-218 refs each) | Stoics (impressions), Schopenhauer | STRONG |
| 4 | **Decision-making & biases** | Kahneman (484 refs), all cognitive books | Gracian (prudence), Confucius (wisdom) | STRONG |
| 5 | **Consciousness & awareness** | Brain/Behavior (232), Gazzaniga (375), Fundamentals (298) | Schopenhauer, Taoism (inner stillness) | STRONG |
| 6 | **Social behavior & empathy** | Behave (257 refs), theory of mind | Confucius (ren/benevolence), Stoics | STRONG |
| 7 | **Aggression & anger** | Behave (155 refs), hormones, amygdala | Seneca (anger letters), Taoism | STRONG |
| 8 | **Addiction & reward/desire** | Biopsychology (52), Psychobiology (51), dopamine | Schopenhauer (will), Stoics (desire mastery) | STRONG |
| 9 | **Fear & mortality** | Fear conditioning, stress response | All Stoics, Schopenhauer, Taoism | STRONG |
| 10 | **Sleep & rest** | Biopsychology (231), Psychobiology (184) | Schopenhauer (rest for mind) | MODERATE |
| 11 | **Reputation & social perception** | Social cognition, implicit bias | Gracian, Schopenhauer (opinion of others) | MODERATE |
| 12 | **Suffering & resilience** | Stress/trauma, adaptation | Schopenhauer (central), Stoics | STRONG |
| 13 | **Human vs animal nature** | Behave (evolution), Cognitive Biology | Schopenhauer (consciousness amplifies suffering) | MODERATE |
| 14 | **Happiness & contentment** | Reward systems, hedonic adaptation | Schopenhauer (absence of pain), Stoics (virtue) | STRONG |
| 15 | **Moral judgment & ethics** | Behave, Determined, Cognitive Neuro | Confucius, Stoics, Gracian | STRONG |

---

## 3. Test Question Design

### Question Design Principles

1. **Open-ended style**: Questions don't explicitly mention "neuroscience AND philosophy" - they ask broadly, testing if RAG finds cross-domain content
2. **Comprehensive ground truth**: Each question has paragraph-length reference answers
3. **Multi-book retrieval**: Each question should require content from 4-6 books
4. **Nuanced complexity**: Questions require synthesizing multiple perspectives, not single-chunk answers

### 15 Comprehensive Evaluation Topics

| # | Topic | What It Tests |
|---|-------|---------------|
| 1 | Free Will & Determinism | Brain determinism vs Stoic fate acceptance |
| 2 | Self-Control & Impulse Regulation | Frontal cortex vs Stoic desire mastery |
| 3 | Emotional Regulation & Tranquility | Limbic system vs Stoic techniques |
| 4 | Decision-Making Biases & Prudence | Kahneman biases vs classical prudence |
| 5 | Consciousness & Self-Awareness | Split-brain research vs philosophical self |
| 6 | Empathy & Social Bonding | Mirror neurons/oxytocin vs benevolence teachings |
| 7 | Anger & Aggression Control | Hormonal aggression vs Seneca on anger |
| 8 | Addiction & The Will | Dopamine/reward vs philosophical will/desire |
| 9 | Fear of Death & Mortality Acceptance | Fear processing vs Stoic death acceptance |
| 10 | Sleep, Rest & Mental Performance | Sleep neuroscience vs wisdom on rest |
| 11 | Reputation, Honor & Self-Perception | Social cognition vs opinion of others |
| 12 | Suffering & Psychological Resilience | Stress biology vs adversity as growth |
| 13 | Human Uniqueness vs Animal Nature | Comparative biology vs philosophical uniqueness |
| 14 | Happiness, Pleasure & Contentment | Reward systems vs philosophical happiness |
| 15 | Moral Judgment & Ethical Behavior | Neuroethics vs virtue ethics |

---

## 4. Human Evaluation Questions (Medium Article)

These 7 questions are designed for human evaluation of chunking and preprocessing strategies, suitable for a Medium article demonstrating RAG capabilities.

### Question 1: Willpower and the Brain

**Question**: *"What is the relationship between willpower and the brain, and why do some people have more self-control than others?"*

**Why It Tests Strategies Well:**
- Requires prefrontal cortex content (neuroscience)
- Requires Stoic self-discipline content (philosophy)
- HyDE should generate "willpower = prefrontal function" hypothesis
- Decomposition should split into brain mechanisms + philosophical practices

**Expected Sources:**
| Domain | Book | Key Content |
|--------|------|-------------|
| Neuroscience | Thinking Fast and Slow | System 1/2, cognitive depletion |
| Neuroscience | Behave | Frontal cortex development, impulse control |
| Neuroscience | Determined | Willpower as deterministic (no free will) |
| Philosophy | The Art of Living (Epictetus) | Control dichotomy, what's in our power |
| Philosophy | The Meditations (Marcus Aurelius) | Self-discipline, morning routines |

### Question 2: Purpose of Suffering

**Question**: *"What is the purpose of suffering in human life, and how should we relate to pain and adversity?"*

**Expected Sources:**
| Domain | Book | Key Content |
|--------|------|-------------|
| Philosophy | Essays and Aphorisms (Schopenhauer) | Suffering as central to existence |
| Philosophy | Wisdom of Life (Schopenhauer) | Contentment through lowered expectations |
| Neuroscience | Biopsychology | Pain processing pathways |
| Philosophy | Letters from a Stoic (Seneca) | Adversity as growth opportunity |
| Neuroscience | Behave | Stress biology, adaptation |

### Question 3: Irrational Decisions

**Question**: *"Why do we make irrational decisions even when we know better, and what can we do about it?"*

**Expected Sources:**
| Domain | Book | Key Content |
|--------|------|-------------|
| Neuroscience | Thinking Fast and Slow | Cognitive biases, System 1 errors |
| Philosophy | The Pocket Oracle (Gracian) | Prudence, avoiding deceit |
| Neuroscience | Behave | Stress affecting judgment |
| Philosophy | The Analects (Confucius) | Wisdom through reflection |

### Question 4: Emotions and Choice

**Question**: *"How do emotions influence our decisions, and should we trust our gut feelings or rely on reason?"*

**Expected Sources:**
| Domain | Book | Key Content |
|--------|------|-------------|
| Neuroscience | Thinking Fast and Slow | Affect heuristic, intuition |
| Neuroscience | Cognitive Neuroscience (Gazzaniga) | Emotional processing pathways |
| Philosophy | The Art of Living (Epictetus) | Impressions and judgment |
| Philosophy | Letters from a Stoic (Seneca) | Passion vs reason |

### Question 5: Anger and Decision-Making

**Question**: *"How does anger affect our judgment, and what are the most effective ways to manage it?"*

**Expected Sources:**
| Domain | Book | Key Content |
|--------|------|-------------|
| Neuroscience | Behave | Aggression circuits, amygdala |
| Neuroscience | Psychobiology of Behaviour | Serotonin and aggression |
| Philosophy | Letters from a Stoic (Seneca) | Anger management letters |
| Philosophy | Tao Te Ching | Soft overcomes hard |

### Question 6: Self-Awareness and Human Uniqueness

**Question**: *"What is self-awareness, how does it arise in the brain, and why do humans have it while most animals don't?"*

**Expected Sources:**
| Domain | Book | Key Content |
|--------|------|-------------|
| Neuroscience | Cognitive Neuroscience (Gazzaniga) | Split-brain, interpreter |
| Neuroscience | Fundamentals of Cognitive Neuroscience | Consciousness states |
| Neuroscience | Cognitive Biology | Evolution of cognition |
| Philosophy | Essays and Aphorisms (Schopenhauer) | Consciousness amplifies suffering |
| Philosophy | Tao Te Ching | Self as construction |

### Question 7: Is the Self an Illusion?

**Question**: *"Is the unified 'self' an illusion created by the brain, and what do ancient philosophers and modern neuroscience agree on about who we really are?"*

**Expected Sources:**
| Domain | Book | Key Content |
|--------|------|-------------|
| Neuroscience | Cognitive Neuroscience (Gazzaniga) | Split-brain interpreter module |
| Neuroscience | Brain and Behavior (Eagleman) | Consciousness construction |
| Philosophy | Essays and Aphorisms (Schopenhauer) | Phenomenal vs noumenal self |
| Philosophy | Tao Te Ching | The real way vs named way |
| Philosophy | The Meditations (Marcus Aurelius) | Cosmic perspective on individual |

---

## 5. Strategy Testing Notes

These questions are designed to:
1. **Test HyDE preprocessing**: Questions with clear hypothetical answers
2. **Test decomposition**: Complex questions that naturally split into sub-questions
3. **Test semantic chunking**: Questions where related content spans multiple books
4. **Test contextual chunking**: Questions where book context improves understanding
5. **Evaluate cross-domain synthesis**: All questions require neuroscience + philosophy integration

**For each question, compare:**
- Retrieval quality (which chunks are retrieved)
- Answer completeness (are both domains represented)
- Source diversity (are multiple books cited)

---

*Consolidated from: evaluation-benchmarks.md, evaluation-questions-analysis.md, human-evaluation-questions.md*
*Last Updated: December 31, 2025*
