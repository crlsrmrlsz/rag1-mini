# Human Evaluation Questions for RAG Testing

These questions are designed for human evaluation of chunking and preprocessing strategies. Each question tests cross-domain retrieval (neuroscience + philosophy) and is suitable for a Medium article demonstrating RAG capabilities.

---

## Question 1: Willpower and the Brain

**Question**: *"What is the relationship between willpower and the brain, and why do some people have more self-control than others?"*

### Why It Tests Strategies Well
- Requires prefrontal cortex content (neuroscience)
- Requires Stoic self-discipline content (philosophy)
- HyDE should generate "willpower = prefrontal function" hypothesis
- Decomposition should split into brain mechanisms + philosophical practices
- Semantic chunking may cluster willpower-related content across books

### Expected Sources
| Domain | Book | Key Content |
|--------|------|-------------|
| Neuroscience | Thinking Fast and Slow | System 1/2, cognitive depletion |
| Neuroscience | Behave | Frontal cortex development, impulse control |
| Neuroscience | Determined | Willpower as deterministic (no free will) |
| Philosophy | The Art of Living (Epictetus) | Control dichotomy, what's in our power |
| Philosophy | The Meditations (Marcus Aurelius) | Self-discipline, morning routines |

### Medium Angle
*"The neuroscience of willpower confirms what the Stoics knew 2,000 years ago - self-control is a muscle that can be trained"*

---

## Question 2: Purpose of Suffering

**Question**: *"What is the purpose of suffering in human life, and how should we relate to pain and adversity?"*

### Why It Tests Strategies Well
- Cross-domain synthesis required (neuroscience of pain + philosophy of suffering)
- Semantic chunking may cluster suffering-related content better
- Contextual chunking adds book context helpful for synthesis
- Decomposition: "What causes suffering?" + "How should we respond?"

### Expected Sources
| Domain | Book | Key Content |
|--------|------|-------------|
| Philosophy | Essays and Aphorisms (Schopenhauer) | Suffering as central to existence |
| Philosophy | Essays, Counsels, Maxims (Schopenhauer) | Happiness as absence of pain |
| Philosophy | Wisdom of Life (Schopenhauer) | Contentment through lowered expectations |
| Neuroscience | Biopsychology | Pain processing pathways |
| Philosophy | Letters from a Stoic (Seneca) | Adversity as growth opportunity |
| Neuroscience | Behave | Stress biology, adaptation |

### Medium Angle
*"Schopenhauer said life is suffering - neuroscience now explains why our brains are wired for dissatisfaction"*

---

## Question 3: Irrational Decisions

**Question**: *"Why do we make irrational decisions even when we know better, and what can we do about it?"*

### Why It Tests Strategies Well
- HyDE: Generates hypothesis about cognitive biases + philosophical prudence
- Decomposition: "What causes irrational decisions?" + "What remedies exist?"
- Semantic chunking: May cluster bias-related content across books
- Tests whether retrieval finds both scientific and practical wisdom

### Expected Sources
| Domain | Book | Key Content |
|--------|------|-------------|
| Neuroscience | Thinking Fast and Slow | Cognitive biases, System 1 errors |
| Philosophy | The Pocket Oracle (Gracian) | Prudence, avoiding deceit |
| Neuroscience | Behave | Stress affecting judgment |
| Philosophy | The Analects (Confucius) | Wisdom through reflection |
| Neuroscience | Brain and Behavior | Prefrontal cortex and decision-making |

### Medium Angle
*"2,000 years of wisdom vs. 50 years of behavioral economics - who understood our flawed thinking better?"*

---

## Question 4: Emotions and Choice

**Question**: *"How do emotions influence our decisions, and should we trust our gut feelings or rely on reason?"*

### Why It Tests Strategies Well
- Tests emotion-decision interaction (limbic + prefrontal)
- Cross-domain: neuroscience of affect heuristic + Stoic impressions theory
- Decomposition: "How do emotions affect decisions?" + "When should we trust intuition?"
- Contextual chunking: Book context helps distinguish perspectives

### Expected Sources
| Domain | Book | Key Content |
|--------|------|-------------|
| Neuroscience | Thinking Fast and Slow | Affect heuristic, intuition |
| Neuroscience | Cognitive Neuroscience (Gazzaniga) | Emotional processing pathways |
| Philosophy | The Art of Living (Epictetus) | Impressions and judgment |
| Philosophy | Letters from a Stoic (Seneca) | Passion vs reason |
| Philosophy | The Meditations (Marcus Aurelius) | The rational self |

### Medium Angle
*"The Stoics said don't trust your first impression - neuroscience explains why they were right"*

---

## Question 5: Anger and Decision-Making

**Question**: *"How does anger affect our judgment, and what are the most effective ways to manage it?"*

### Why It Tests Strategies Well
- Specific emotion focus enables precise retrieval testing
- Cross-domain: amygdala/cortisol + Seneca's "On Anger"
- HyDE: Should generate anger-management hypothesis spanning domains
- Good for testing if semantic search clusters anger-related content

### Expected Sources
| Domain | Book | Key Content |
|--------|------|-------------|
| Neuroscience | Behave | Aggression circuits, amygdala |
| Neuroscience | Psychobiology of Behaviour | Serotonin and aggression |
| Philosophy | Letters from a Stoic (Seneca) | Anger management letters |
| Philosophy | Tao Te Ching | Soft overcomes hard |
| Philosophy | The Meditations (Marcus Aurelius) | Controlling reactions |

### Medium Angle
*"Seneca wrote about anger 2,000 years ago - neuroscience now proves he was onto something"*

---

## Question 6: Self-Awareness and Human Uniqueness

**Question**: *"What is self-awareness, how does it arise in the brain, and why do humans have it while most animals don't?"*

### Why It Tests Strategies Well
- Requires integration of consciousness research + philosophical self
- Semantic chunking: May cluster self-awareness content across domains
- Decomposition: "What brain mechanisms create self-awareness?" + "What makes human self-awareness unique?"
- Tests retrieval of evolutionary/comparative content

### Expected Sources
| Domain | Book | Key Content |
|--------|------|-------------|
| Neuroscience | Cognitive Neuroscience (Gazzaniga) | Split-brain, interpreter |
| Neuroscience | Fundamentals of Cognitive Neuroscience | Consciousness states |
| Neuroscience | Cognitive Biology | Evolution of cognition |
| Philosophy | Essays and Aphorisms (Schopenhauer) | Consciousness amplifies suffering |
| Philosophy | Tao Te Ching | Self as construction |

### Medium Angle
*"The split-brain experiments that shattered our understanding of the self"*

---

## Question 7: Is the Self an Illusion?

**Question**: *"Is the unified 'self' an illusion created by the brain, and what do ancient philosophers and modern neuroscience agree on about who we really are?"*

### Why It Tests Strategies Well
- Deepest cross-domain synthesis required
- HyDE: Should generate hypothesis about constructed self
- Contextual chunking: Book context crucial for distinguishing perspectives
- Tests whether system finds convergent ideas across very different traditions

### Expected Sources
| Domain | Book | Key Content |
|--------|------|-------------|
| Neuroscience | Cognitive Neuroscience (Gazzaniga) | Split-brain interpreter module |
| Neuroscience | Brain and Behavior (Eagleman) | Consciousness construction |
| Philosophy | Essays and Aphorisms (Schopenhauer) | Phenomenal vs noumenal self |
| Philosophy | Tao Te Ching | The real way vs named way |
| Philosophy | The Meditations (Marcus Aurelius) | Cosmic perspective on individual |

### Medium Angle
*"Both the Buddha and neuroscience say the self is an illusion - here's why that's liberating, not terrifying"*

---

## Recommended Combinations for Articles

| Focus | Questions | Rationale |
|-------|-----------|-----------|
| Decision-Making | Q3 + Q4 | Irrational decisions + emotions |
| Consciousness | Q6 + Q7 | Self-awareness + self as illusion |
| Practical Wisdom | Q1 + Q5 | Willpower + anger management |
| Existential | Q2 + Q7 | Suffering + self illusion |
| Maximum Variety | Q2 + Q7 | Covers suffering and consciousness |

---

## Testing Notes

These questions are designed to:
1. **Test HyDE preprocessing**: Questions with clear hypothetical answers
2. **Test decomposition**: Complex questions that naturally split into sub-questions
3. **Test semantic chunking**: Questions where related content spans multiple books
4. **Test contextual chunking**: Questions where book context improves understanding
5. **Evaluate cross-domain synthesis**: All questions require neuroscience + philosophy integration

For each question, compare:
- Retrieval quality (which chunks are retrieved)
- Answer completeness (are both domains represented)
- Source diversity (are multiple books cited)
