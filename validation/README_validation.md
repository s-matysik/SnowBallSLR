# Cross-vendor LLM screening validation

`llm_screen_validate.py` reworks `ewaluator.ipynb` into a script you run one design at a time.

## Setup

For a numbered walkthrough with costs and what to read in each
output, see `RUN_ME.md`. This file is the reference; that one is the procedure.

    pip install openai anthropic google-genai

Put your keys in the environment, or in a `.env` file beside the script (it is gitignored):

    OPENAI_API_KEY=...
    ANTHROPIC_API_KEY=...
    GOOGLE_API_KEY=...
    DEEPSEEK_API_KEY=...
    XAI_API_KEY=...

Any subset works — vendors without a key are skipped and the run says which it used. **The keys
that were pasted into `ewaluator.ipynb` should be treated as compromised and rotated**; they are in
a file on disk and in that notebook's history.

## The six designs

Ready-made inputs are supplied, so each command below runs as written. `validation_records.json` holds 311 records from the blockchain × customer loyalty case: 296 candidates stratified across the
whole score range plus all 15 seeds mixed in unmarked, so blind controls work on the same basis as
the candidates.

**Step 0a — confirm which vendors are usable.** No API calls, no cost:

    python validation/llm_screen_validate.py check-keys

It reports how many distinct vendor *families* your keys span. Fewer than two and the agreement
design measures within-family consistency, which is the limitation this whole exercise exists to
remove.

**Step 0b — score once, validate many times.** Scoring costs money; the validations do not.

    python validation/llm_screen_validate.py score \
      --records validation/validation_records.json \
      --question cases/research_question_mid.json \
      --out cases/xvendor_scores.json

**1. Agreement, within vs between vendor families** — the design that answers the reviewer.

    python validation/llm_screen_validate.py agreement \
      --scores cases/xvendor_scores.json --out cases/xvendor_agreement.json

Reports ICC within each family and correlation between families. On the existing three-case scores
it correctly reports one family and warns that within-family agreement is not validation.

**2. Blind positive controls**

    python validation/llm_screen_validate.py controls \
      --scores cases/xvendor_scores.json \
      --seeds cases/seed_selection.json --case mid --out cases/xvendor_controls.json

**3. Threshold sensitivity band**

    python validation/llm_screen_validate.py threshold \
      --scores cases/xvendor_scores.json --out cases/xvendor_threshold.json

**4. Wording and order robustness** — three scoring passes, so roughly 3× the cost of step 0.

    python validation/llm_screen_validate.py robustness \
      --records validation/validation_records.json \
      --question cases/research_question_mid.json \
      --out cases/xvendor_robustness.json --limit 150

**5. Negative control: a plausible but wrong question** — two passes.

    python validation/llm_screen_validate.py negative \
      --records validation/validation_records.json \
      --question cases/research_question_mid.json \
      --wrong-question validation/wrong_question_for_negative_control.json \
      --out cases/xvendor_negative.json --limit 150

**6. Convergent validity against non-LLM bibliometric signals** — free, no API calls.

    python validation/llm_screen_validate.py bibliometric \
      --scores cases/xvendor_scores.json \
      --records validation/validation_records.json \
      --seed-records validation/validation_seeds.json \
      --out cases/xvendor_bibliometric.json

Designs 1–3 and 6 read one scores file and cost nothing. Designs 4–5 must score again, because they
change the prompt.

## What each design proves, and what it does not

| Design | Answers | Does not answer |
|---|---|---|
| Agreement (between families) | do independent models converge | whether they match a human |
| Blind controls | does the screener recognise known-relevant work | whether it responds to the *question* |
| Threshold band | how much the included set depends on the cut | whether the ranking is right |
| Robustness | is the screener stable under rewording and reordering | whether the criteria are the right ones |
| Negative control | is it responding to the question, not to prose quality | how it behaves on borderline records |
| Bibliometric | does an independent, non-model signal agree | whether either signal is correct |

**Design 6 is the only one whose signal does not come from a language model**, which makes it the
cheapest independent check available and the one to run first. Read the *pattern*: topical predictors
above chance with prestige predictors at chance is evidence the screener judges relevance. Prestige
predictors above the topical ones would mean the scores track citation counts or venue standing —
a failure no positive control detects, because a prestige-driven screener also scores seeds highly.

None of the six is a substitute for double-screening by an independent reviewer — they bound the
screener rather than validate it against expert judgement. But an author-versus-ensemble coefficient
is not that substitute either: the author who wrote the rubric and chose the seeds cannot serve as
the independent reviewer, so such a κ measures fidelity to the framing the models were handed. The
manuscript therefore reports the six designs and no human coefficient. `score_human_agreement.py` is
here for reviews where a colleague who did not author the protocol can screen the blind sample.

## Two design decisions worth knowing

**Common-basis guard.** Every statistic is computed only over records scored by *every* rater in
the comparison, and the count is printed. Mixing records scored by one model with records scored by
three yields a number that cannot be reproduced later — that happened in the three-case study,
where one case's reported control AUC of 0.781 could not be recovered from its own saved scores
(three plausible bases gave 0.58, 0.61 and 0.73). The guard makes every figure checkable.

**Failed batches are recorded, not dropped.** A batch that will not parse is retried once with a
larger token budget; whatever still fails is written to `failures` and excluded from the common
basis. A silently dropped record is an invisible hole in every later statistic.
