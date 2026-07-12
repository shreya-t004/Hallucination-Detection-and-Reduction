# Experimental Figures

This directory contains the figures generated from the final
Internal Cross-Examination experiments conducted on the complete TruthfulQA
validation split.

## Figure 1 — Baseline vs ICE Hallucination Rate

**File:** `fig1_baseline_vs_ice.png`

Compares the hallucination rate produced by each Qwen2.5 model before and
after applying Internal Cross-Examination.

The figure shows that ICE produced meaningful reductions for the 1.5B and
3B models, while the 0.5B model showed minimal improvement.

---

## Figure 2 — Fix-to-Break Analysis

**File:** `fig2_fix_break.png`

Compares the number of hallucinated answers corrected by ICE with the number
of initially correct answers damaged during revision.

The fix-to-break ratio measures how many answers were corrected for every
answer that was broken.

---

## Figure 3 — Model Scaling Behaviour

**File:** `fig3_scaling.png`

Shows how the effectiveness of Internal Cross-Examination changes as model
parameter size increases.

The results indicate that self-verification becomes more effective once the
model has sufficient reasoning and instruction-following capability.

---

## Figure 4 — Hallucination Type Distribution

**File:** `fig4_types.png`

Displays the distribution of hallucination categories identified during the
experiments.

The evaluated categories include:

- Fabrication
- Confusion
- Reasoning error
- Outdated information

---

## Figure 5 — Evaluation Score Distribution

**File:** `fig5_distribution.png`

Visualizes the distribution of evaluation scores used to classify generated
answers.

The scoring framework combines Natural Language Inference and semantic
similarity rather than relying on lexical overlap alone.

---

## Figure 6 — Baseline-to-ICE Answer Movements

**File:** `fig6_movements.png`

Shows how answer classifications changed after applying ICE.

The movement categories include:

- Correct to correct
- Hallucinated to correct
- Correct to hallucinated
- Hallucinated to hallucinated

This figure highlights both the corrections produced by ICE and the risks
introduced during the revision stage.

---

## Experimental Context

The figures summarize experiments conducted using:

- TruthfulQA validation split
- 817 questions
- Qwen2.5-0.5B-Instruct
- Qwen2.5-1.5B-Instruct
- Qwen2.5-3B-Instruct
- Balanced NLI and semantic-similarity evaluation
- Google Colab with an NVIDIA A100 GPU

The complete implementation and raw experimental outputs are maintained
privately.
