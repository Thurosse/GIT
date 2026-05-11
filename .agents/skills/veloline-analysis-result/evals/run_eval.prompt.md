---
name: run-eval-analysis-result
description: "Evaluate trigger-routing accuracy for the veloline-analysis-result skill. Run when asked to 'eval the analysis skill', 'score the analysis-result trigger eval', or 'check if the analysis skill is invoked correctly'."
agent: agent
---

# Eval: veloline-analysis-result trigger routing

You are evaluating whether the `veloline-analysis-result` skill is correctly triggered (or correctly skipped) for each test case in [cases.yml](./cases.yml).

## Skill routing rules (from the skill's description)

**Invoke** when the user explicitly asks to produce velocity-pipeline analysis figures — including but not limited to:
- "run veloline analysis stage"
- "regenerate velocity plots"
- "run shift analysis"
- "produce fit-quality figures"
- "make co-expression plots from latest run"
- Any clear synonym, contextual paraphrase, or downstream trigger that unambiguously requests stage-3 execution.

**Skip (do NOT invoke)** when:
- The user asks for an ad-hoc plot tweak inside the notebook (e.g., change colormap, adjust axis limits).
- The user asks for matplotlib styling help (formatting, colours, colorbars) without requesting a full pipeline run.
- The user proposes a new analytical experiment unrelated to the existing pipeline (e.g., a new metric, a custom analysis).
- The user asks to run fit1 or fit2 / the inference stage (routes to `veloline-run-inference`).
- The user asks to rebuild mp or run setup (routes to `veloline-pipeline-setup`).
- The user is asking a conceptual or explanatory question (no stage execution needed).
- The user wants to edit source code (e.g., `velocity.py`, `shift.py`, `coexpression.py`).

## Evaluation procedure

1. Read all cases from [cases.yml](./cases.yml).
2. For **each case**, reason step by step:
   a. What is the user's intent in the prompt?
   b. Does it match any invoke trigger? Does it match any skip condition?
   c. What is your **predicted** routing: `invoke` or `skip`?
   d. Does your prediction match `expected`?
3. Produce a results table (see format below).
4. Print a summary line with the overall score.
5. For every **FAIL**, write a one-sentence diagnosis explaining why the routing is non-obvious.

## Output format

```
## Results

| ID | Category | Prompt (truncated) | Expected | Predicted | Result |
|----|----------|--------------------|----------|-----------|--------|
| p01_exact_run_analysis_stage | P-exact | "run veloline analysis stage" | invoke | invoke | PASS |
| ... | ... | ... | ... | ... | ... |

## Score: N / 18 cases passed

## Failures
- <id>: <one-sentence diagnosis>
```

Start evaluating now.
