---
name: run-eval-inference
description: "Evaluate trigger-routing accuracy for the veloline-run-inference skill. Run when asked to 'eval the inference skill', 'score the run-inference trigger eval', or 'check if the inference skill is invoked correctly'."
agent: agent
---

# Eval: veloline-run-inference trigger routing

You are evaluating whether the `veloline-run-inference` skill is correctly triggered (or correctly skipped) for each test case in [cases.yml](./cases.yml).

## Skill routing rules (from the skill's description)

**Invoke** when the user explicitly asks to run the velocity pipeline's inference stage — including but not limited to:
- "run veloline inference stage"
- "run fit1"
- "run fit2"
- "execute SVI on the velocity model"
- "fit the Pyro models"
- "rerun the variational inference"
- Any clear synonym, contextual paraphrase, or downstream trigger that unambiguously requests stage-2 execution.

**Skip (do NOT invoke)** when:
- The user is editing model definitions in `models.py` (code-edit, not execution).
- The user is asking a conceptual or explanatory question about Pyro, SVI, MCMC, or the ELBO.
- The user is working on plots or diagnostics (routes to `veloline-analysis-result`).
- The user asks to run the setup / rebuild mp (routes to `veloline-pipeline-setup`).
- The user wants to regenerate plots from saved posteriors (routes to `veloline-analysis-result`).
- The user wants to edit any source file (`fit2.py`, `models.py`, `metaparams.py`, etc.).

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
| p01_exact_run_inference_stage | P-exact | "run veloline inference stage" | invoke | invoke | PASS |
| ... | ... | ... | ... | ... | ... |

## Score: N / 18 cases passed

## Failures
- <id>: <one-sentence diagnosis>
```

Start evaluating now.
