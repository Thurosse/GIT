---
name: run-eval-pipeline-setup
description: "Evaluate trigger-routing accuracy for the veloline-pipeline-setup skill. Run when asked to 'eval the setup skill', 'score the pipeline-setup trigger eval', or 'check if the setup skill is invoked correctly'."
agent: agent
---

# Eval: veloline-pipeline-setup trigger routing

You are evaluating whether the `veloline-pipeline-setup` skill is correctly triggered (or correctly skipped) for each test case in [cases.yml](./cases.yml).

## Skill routing rules (from the skill's description)

**Invoke** when the user explicitly asks to run the velocity pipeline's setup stage — including but not limited to:
- "run veloline pipeline setup"
- "rebuild mp"
- "prepare data for fit1/fit2"
- "execute setup stage of the velocity pipeline"
- "regenerate adata_fit"
- Any clear synonym, contextual paraphrase, or downstream trigger that unambiguously requests stage-1 execution.

**Skip (do NOT invoke)** when:
- The user asks to modify `metaparams.py` (they should edit it themselves first).
- The user asks a plotting or visualisation question (routes to `veloline-analysis-result`).
- The user asks to run fit1 or fit2 / the inference stage (routes to `veloline-run-inference`).
- The user is asking a conceptual or explanatory question (no stage execution needed).
- The user wants to edit source code (e.g., `data_loading.py`, `models.py`).

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
| p01_exact_run_setup | P-exact | "run veloline pipeline setup" | invoke | invoke | PASS |
| ... | ... | ... | ... | ... | ... |

## Score: N / 18 cases passed

## Failures
- <id>: <one-sentence diagnosis>
```

Start evaluating now.
