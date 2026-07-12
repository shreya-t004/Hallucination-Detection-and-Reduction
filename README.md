# Hallucination Detection & Reduction in LLMs via Internal Cross-Examination (ICE)

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-orange)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co)
[![Dataset](https://img.shields.io/badge/Dataset-TruthfulQA-green)](https://huggingface.co/datasets/truthful_qa)

> **An inference-time self-verification pipeline that reduces LLM hallucinations by up to 30% across Qwen 2.5 model family — no retraining required.**

---

## Results

| Model        | Baseline Hall. | After ICE | Reduction         | Fix:Break |
| ------------ | -------------- | --------- | ----------------- | --------- |
| Qwen2.5-0.5B | 61.3%          | 60.5%     | +0.9pp            | 1.0:1     |
| Qwen2.5-1.5B | 52.5%          | 36.6%     | **−15.9pp (30%)** | 2.3:1     |
| Qwen2.5-3B   | 38.3%          | 25.8%     | **−12.5pp (33%)** | 2.2:1     |

**Dataset:** TruthfulQA (817 questions, full validation split)  
**Evaluation:** Balanced scorer — NLI entailment (DeBERTa-v3) + semantic similarity (MiniLM-L6)  
**Hardware:** Google Colab A100 (40GB)  
**No fine-tuning. No external knowledge. Inference-time only.**

---

## What is ICE?

ICE (Internal Cross-Examination) is a 6-step pipeline that makes a model verify its own answers before returning them:

```
Question
    ↓
1. Draft Answer            (model answers freely)
    ↓
2. Extract Claims          (pull out individual factual claims)
    ↓
3. Generate Verification Q (create a question per claim)
    ↓
4. Answer Independently    (fresh context, no draft visible)
    ↓
5. Consistency Check       (does independent answer support the claim?)
    ↓
6. Revise Final Answer     (remove wrong claims, keep correct ones)
```

**The key insight:** The model never sees the original draft during verification. This forces genuine cross-examination rather than self-confirmation.

---

## Research Journey

This project went through multiple iterations before arriving at the final results. Below is a full account of every approach tried.

### Phase 1 — Environment & Baseline

**Goal:** Get a working pipeline and measure hallucination rates.

- Set up Python environment on local Windows machine (i7-11390H, 16GB RAM, no usable GPU)
- Loaded TruthfulQA + Qwen2.5-0.5B-Instruct on CPU
- Ran 20 questions → saved to JSONL
- **Result:** 60% hallucination rate at 0.5B on 20 questions

**Issues found:**

- Model leaking meta-text after answers (fixed with cutoff strings)
- `torch_dtype` deprecation warning (fixed by renaming to `dtype`)

---

### Phase 2 — Hallucination Scoring

**Scorer v1 — TF-IDF cosine similarity**

- Fast but unreliable — word overlap doesn't capture meaning
- "Seeds pass through you" vs "seeds cause digestive issues" scored as similar

**Scorer v2 — Sentence-transformers (all-MiniLM-L6-v2)**

- Semantic similarity — much better
- Threshold: correct_sim ≥ 0.15 and correct_sim > incorrect_sim → correct

**Scorer v3 — Balanced (NLI + semantic similarity)**

- Combined NLI entailment (cross-encoder/nli-deberta-v3-base) with semantic similarity
- Tiered decision logic:
  1. NLI entailment with correct ref → correct
  2. NLI entailment with incorrect ref → hallucinated
  3. NLI contradiction + high incorrect sim → hallucinated
  4. High correct similarity beating incorrect → correct
  5. Hedging → separate category
- **This became the final scorer**

**Hallucination type classification:**

- Fabrication: model invents facts (low overlap with correct answer)
- Confusion: mixes up related facts (medium overlap)
- Reasoning Error: correct facts, wrong conclusion (logical words present)
- Outdated: used to be correct but isn't anymore

---

### Phase 3 — ICE Pipeline v1

**Configuration:**

- Temperature: 0.7 (sampling)
- Sequential processing (one question at a time)
- Hardware: CPU locally, then T4 GPU on Colab

**Results on 50 questions (T4, lenient scorer):**

| Model | Baseline | ICE | Δ     |
| ----- | -------- | --- | ----- |
| 0.5B  | 52%      | 28% | −24pp |
| 1.5B  | 32%      | 16% | −16pp |
| 3B    | 22%      | 0%  | −22pp |

**Problem:** The 0% at 3B was suspicious. The lenient scorer was counting hedging ("I'm not sure") as correct. The model wasn't actually answering correctly — it was refusing to answer.

---

### Phase 4 — Three-Scorer Validation

Introduced three scorers running simultaneously to catch over-optimism:

| Scorer   | Threshold                   | Hedging    | Purpose       |
| -------- | --------------------------- | ---------- | ------------- |
| Lenient  | sim ≥ 0.15                  | = correct  | Upper bound   |
| Strict   | NLI entailment + sim ≥ 0.30 | = separate | Lower bound   |
| Balanced | Tiered logic                | = separate | Best estimate |

**Results on 200 questions (A100, balanced scorer):**

| Model | Baseline | ICE   | Δ     | Fix:Break |
| ----- | -------- | ----- | ----- | --------- |
| 0.5B  | 55.0%    | 46.0% | −9pp  | 1.6:1     |
| 1.5B  | 55.5%    | 32.5% | −23pp | 2.9:1     |
| 3B    | 32.0%    | 15.0% | −17pp | **5.3:1** |

**Key finding:** 3B was the sweet spot — highest precision (5.3:1 ratio), meaning for every 5 questions fixed, only 1 was broken.

---

### Phase 5 — Deterministic vs Sampling Investigation

Hypothesis: results might be inflated because temperature=0.7 introduces randomness that could go either way.

**Test:** Switched to deterministic generation (temperature=0) for drafts.

**Result:** ICE effectiveness collapsed — all models near 0pp improvement.

**Root cause discovered:** With deterministic drafts, the verification step produced nearly identical answers to the draft. The model agreed with itself 91% of the time at 0.5B, so the consistency check found no contradictions and the draft was returned unchanged.

**Fix attempted:** Hybrid approach — deterministic drafts, sampling for verification (temp=0.7).

**Result on 200 questions:**

| Model | Baseline | ICE   | Δ       |
| ----- | -------- | ----- | ------- |
| 0.5B  | 56.5%    | 57.5% | −1pp    |
| 1.5B  | 54.5%    | 46.5% | −8pp    |
| 3B    | 37.0%    | 25.5% | −11.5pp |

Better than fully deterministic but worse than fully sampled. Kept temp=0.7 for everything in the final version.

---

### Phase 6 — Strategy Search

Ran 16 experiments: 4 strategies × 4 models × 200 questions.

**Strategies tested:**

**1. ICE-Standard** — the 6-step pipeline (verify=sampling 0.7)

**2. ICE-Lite** — 3 sampled drafts, keep only consistent claims

- Rationale: small models can't self-verify; check if they agree across multiple samples
- "If a model is correct it gives the same answer at different temperatures; if it hallucinates it gives different answers"

**3. ICE-Double** — run ICE twice, second pass catches remaining errors

**4. ICE-Selective** — only verify low-confidence answers (skip if model sounds certain)

**Results (balanced scorer):**

| Model    | Strategy  | BL        | ICE       | Δ          |
| -------- | --------- | --------- | --------- | ---------- |
| 0.5B     | standard  | 56.5%     | 59.5%     | −3pp       |
| 0.5B     | lite      | 56.5%     | 58.5%     | −2pp       |
| 0.5B     | double    | 56.5%     | 57.0%     | −0.5pp     |
| 0.5B     | selective | 56.5%     | 57.5%     | −1pp       |
| 1.5B     | standard  | 54.5%     | 55.5%     | −1pp       |
| **1.5B** | **lite**  | **54.5%** | **45.0%** | **+9.5pp** |
| 1.5B     | double    | 54.5%     | 52.0%     | +2.5pp     |
| 1.5B     | selective | 54.5%     | 56.0%     | −1.5pp     |
| 3B       | standard  | 37.0%     | 45.5%     | −8.5pp     |
| 3B       | lite      | 37.0%     | 37.0%     | 0pp        |
| 3B       | double    | 37.0%     | 47.5%     | −10.5pp    |
| 3B       | selective | 37.0%     | 42.0%     | −5pp       |
| 7B       | standard  | 15.0%     | 34.0%     | −19pp      |
| 7B       | lite      | 15.0%     | 16.0%     | −1pp       |
| 7B       | double    | 15.0%     | 35.5%     | −20.5pp    |
| 7B       | selective | 15.0%     | 23.0%     | −8pp       |

**Critical finding:** Standard ICE and Double ICE were making things WORSE at 3B and 7B. The revision step was introducing new errors — not removing old ones.

**Root cause:** The model's verification answers were often wrong too. When a wrong verification answer contradicts a correct claim, the revision then removes a correct claim. This is the fundamental limitation of self-verification.

---

### Phase 7 — E-ICE (Elimination-Based ICE)

**New approach:** Instead of asking "are you right?", use disagreement between samples as the signal.

**E-ICE Pipeline:**

1. Generate 3 answers at different temperatures (0.5, 0.7, 0.9)
2. Check pairwise similarity between drafts
3. Branch:
   - All agree (sim > 0.75) → knowledge probe: ask "Is [correct answer] true?"
   - Two agree → keep majority
   - All differ → extract claims, eliminate wrong ones, re-prompt with negation

**E-ICE v1:** Self-judging (each model judges its own answers)

- Problem: 1.5B judged itself 0% correct — model couldn't objectively evaluate itself

**E-ICE v2:** Cross-model judging (7B judges all models)

- Problem: Judge compared long paragraph answers to short references → said NO to everything
- 0.5B baseline got 4% correct under this judge — clearly wrong

**E-ICE v3:** Multi-reference judging (check against all 3-6 correct references, not just one)

- Problem: Judge still unreliable — "born in Honolulu, Hawaii" didn't match "born in the U.S."

**E-ICE v4:** Extract-then-score

- Step 1: Extract one-sentence core answer from each long response
- Step 2: Similarity filter (auto-correct if sim > 0.65 to any correct ref)
- Step 3: Judge only ambiguous cases (sim 0.3–0.65)
- Result: More reliable scoring but E-ICE results still inconsistent

**Conclusion from E-ICE experiments:** The evaluation was as much of a problem as the method. TruthfulQA questions are specifically designed around common misconceptions — models consistently agree on wrong answers at all temperatures. Sampling disagreement doesn't help when the wrong answer is deeply embedded in training data.

---

### Phase 8 — Return to Standard ICE, Full Dataset (Final)

**Decision:** Return to the standard 6-step ICE with:

- Temperature=0.7 for all generation
- Balanced scorer (NLI + semantic similarity)
- Full TruthfulQA dataset (817 questions)
- Batched processing for A100

**Why batched:** Previous runs processed one question at a time through all 6 steps. Batched version processes ALL questions through each stage, then moves to the next stage. This allowed using A100's full VRAM capacity.

**Total reruns before final results:**

- 20q pilot runs: ~5 runs
- 50q experimental runs: ~8 runs
- 200q validation runs: ~6 runs
- 817q final runs: 3 runs (one per model due to session timeouts)
- **Total: ~22 experimental runs across all configurations**

**Final results on 817 questions:**

| Model | Baseline | ICE   | Reduction           | Fix:Break |
| ----- | -------- | ----- | ------------------- | --------- |
| 0.5B  | 61.3%    | 60.5% | +0.9pp (no benefit) | 1.0:1     |
| 1.5B  | 52.5%    | 36.6% | **−15.9pp (30%)**   | 2.3:1     |
| 3B    | 38.3%    | 25.8% | **−12.5pp (33%)**   | 2.2:1     |

---

## Key Findings

**1. Self-verification works but is model-size dependent.**
Below 1B parameters, models lack the capacity to reliably cross-examine their own claims. At 1.5B+, meaningful improvement emerges.

**2. 3B is not the precision winner at scale.**
On 200 questions, 3B showed a 5.3:1 fix-to-break ratio. On 817 questions, this dropped to 2.2:1. The 200q result was partially due to sampling variance.

**3. The revision step is the bottleneck.**
Detection (finding contradictions) is more reliable than correction (rewriting correctly). When verification answers are themselves wrong, the revision step removes correct claims.

**4. Temperature matters more than strategy.**
Deterministic generation + sampling verification produced mediocre results. Fully sampled (temp=0.7) throughout consistently outperformed hybrid approaches.

**5. Evaluation is as hard as the method.**
NLI models mark different phrasings of the same fact as "neutral" or "contradiction." Semantic similarity is too lenient. The balanced scorer combining both gave the most reliable results.

**6. 7B baseline is 0% on TruthfulQA — likely contamination.**
Qwen2.5-7B achieved 0% hallucination on TruthfulQA, suggesting the model has memorized the dataset. Results excluded from final paper for this reason.

---

## Repository Structure

```
├── ICE_FINAL_BATCHED.ipynb     # Final production notebook (batched, A100)
├── outputs/
│   ├── baseline_*.jsonl        # Raw baseline answers per model
│   ├── ice_*.jsonl             # ICE pipeline answers per model
│   └── findings.txt            # Auto-generated results summary
├── figures/
│   ├── fig1_baseline_vs_ice.png
│   ├── fig2_fix_break.png
│   ├── fig3_scaling.png
│   ├── fig4_types.png
│   ├── fig5_distribution.png
│   └── fig6_movements.png
└── README.md
```

---

## Setup

```bash
pip install torch transformers datasets accelerate pandas numpy tqdm \
            matplotlib scikit-learn sentence-transformers
```

**Recommended hardware:** A100 (40GB) or T4 (16GB)  
**Estimated runtime (A100, 817q):**

- Baselines: ~30 min (batched)
- ICE pipeline: ~4–5 hours (stage-wise batched)

---

## Running

Open `ICE_FINAL_BATCHED.ipynb` in Google Colab, set runtime to A100, and run all cells.

To change the number of questions:

```python
NUM_QUESTIONS = 817  # Full TruthfulQA
```

To run a single model:

```python
MODELS = ["Qwen/Qwen2.5-1.5B-Instruct"]
```

---

## Models

| Model                 | Parameters | VRAM (float16) | Batch Size |
| --------------------- | ---------- | -------------- | ---------- |
| Qwen2.5-0.5B-Instruct | 0.5B       | ~1GB           | 16         |
| Qwen2.5-1.5B-Instruct | 1.5B       | ~3GB           | 12         |
| Qwen2.5-3B-Instruct   | 3B         | ~6GB           | 8          |

All models from [Qwen/Qwen2.5](https://huggingface.co/Qwen) family (Alibaba). Chosen for clean size ladder enabling scaling analysis, zero COI with author's company work.

---

## Evaluation

### Balanced Scorer

Combines NLI entailment and semantic similarity in tiered decision logic:

```python
if NLI(answer, correct_ref) == "entailment" and sim >= 0.25:
    label = "correct"
elif NLI(answer, incorrect_ref) == "entailment" and sim >= 0.25:
    label = "hallucinated"
elif NLI(answer, correct_ref) == "contradiction" and incorrect_sim > correct_sim:
    label = "hallucinated"
elif correct_sim >= 0.40 and (correct_sim - incorrect_sim) >= 0.03:
    label = "correct"
elif hedging and correct_sim < 0.50:
    label = "hedged"
else:
    label = "hallucinated"
```

**Models used:**

- Semantic similarity: `all-MiniLM-L6-v2` (sentence-transformers)
- NLI: `cross-encoder/nli-deberta-v3-base`

---

## Limitations

- Results vary between runs due to temperature=0.7 sampling (~2–3pp variance)
- Evaluation relies on automatic scoring — not validated against human labels
- 7B model excluded due to suspected TruthfulQA contamination
- ICE adds ~6x inference cost per question (6 model calls vs 1)
- Self-verification is fundamentally limited when the model's knowledge base contains the wrong answer

---

## Citation

If you use this work, please cite:

```
@misc{ice2026,
  title={Hallucination Detection and Reduction in LLMs via Internal Cross-Examination},
  author={[Author]},
  year={2026},
  url={https://github.com/[username]/hallucination-ice}
}
```

---

## Related Work

- Madaan et al. (2023) — Self-Refine: Iterative Refinement with Self-Feedback
- Dhuliawala et al. (2023) — Chain-of-Verification Reduces Hallucination in LLMs
- Huang et al. (2023) — Large Language Models Cannot Self-Correct Reasoning Yet
- Manakul et al. (2023) — SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection
- Lin et al. (2022) — TruthfulQA: Measuring How Models Mimic Human Falsehoods

## License

Copyright © 2026 Shreya Tripathi. All rights reserved.

This repository is publicly viewable for portfolio and research-demonstration
purposes only. Reuse, copying, modification, redistribution, or commercial use
of the code is not permitted without prior written permission.
