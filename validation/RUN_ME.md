# Step by step: running the validations against live APIs

Everything below is copy-paste. Run from the repository root
(`cd "/Users/sebastianmatysik/!!!SnowBallSLR"`), and use the repo's own interpreter
`.venv/bin/python` so the library and its dependencies resolve.

---

## Step 0 — rotate the exposed keys first

`ewaluator.ipynb` has six live API keys pasted into its cells. Those keys are on disk and in the
notebook's saved history. **Revoke and reissue all six before you use any of them here.** Nothing
below needs the old keys; it needs new ones.

---

## Step 1 — install the vendor SDKs

    .venv/bin/pip install openai anthropic google-genai

Only these three packages are needed. DeepSeek and xAI speak the OpenAI protocol, so the `openai`
package covers them.

---

## Step 2 — put the new keys in `.env`

Create `.env` in the repository root. It is already in `.gitignore`, so it will never be committed:

    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    GOOGLE_API_KEY=...
    DEEPSEEK_API_KEY=sk-...
    XAI_API_KEY=xai-...

Any subset works. **Two different vendors is the minimum that makes this exercise worth running** —
the whole point is to escape the single-family limitation. Six is what the shipped registry covers.

Two setup facts worth knowing, both found the hard way:

- **Moonshot runs two platforms with separate key namespaces.** `api.moonshot.cn` (mainland) and
  `api.moonshot.ai` (international) do not share keys: a key issued for one returns
  `401 Invalid Authentication` on the other. The model catalogues differ too —
  `kimi-k2-turbo-preview` exists on `.cn` and 404s on `.ai`. The registry ships with the
  international endpoint and `kimi-k2.6`. If your key 401s, it is a mainland key: switch
  `base_url` to `https://api.moonshot.cn/v1` and the model to `kimi-k2-turbo-preview` in the
  `VENDORS` tuple.
- **DeepSeek and xAI speak the OpenAI protocol**, so the `openai` package covers them; only
  Anthropic and Google need their own SDKs.

Check what the script sees. This makes no API calls and costs nothing:

    .venv/bin/python validation/llm_screen_validate.py check-keys

It lists every vendor, whether its key is set, and how many distinct families they span. If it
reports fewer than two families it says so explicitly — add another vendor before continuing, because
one family is exactly the limitation this exercise exists to remove.

---

## Step 3 — which data to use

**Use `validation/validation_records.json`.** It is the only records file you need, and it is built
so all six designs run on it:

- **311 records** from the blockchain × customer loyalty review (the best-validated case, ICC 0.827,
  so any degradation you see is attributable to the manipulation and not to a noisy baseline)
- **296 candidates stratified across the whole 1–10 score range** — 60 per band. Stratification
  matters: a plain random sample would be dominated by obvious excludes and would inflate every
  agreement statistic
- **all 15 seed records mixed in, unmarked and shuffled**, so the blind controls in design 2 are
  scored on exactly the same basis as the candidates
- every record carries the fields design 6 needs: `cited_by_count` and `reference_count` are present
  for all 311, `venue` for 303, `abstract` for 307. Note that **42 records have `cited_by_count` = 0**
  — that is a real measured value, not a gap (OpenAlex reports 0 for each of the six spot-checked),
  so uncited papers are correctly treated as uncited rather than dropped. The script drops a record
  from a predictor's AUC only when that predictor's value is genuinely absent, and reports the counts
  it used under `auc_class_counts`; read `n_missing_field` there rather than inferring coverage from
  the class sizes.

Supporting files, all already present:

| File | Used by | What it is |
|---|---|---|
| `validation/validation_records.json` | all designs | the 311 records |
| `validation/validation_seeds.json` | design 6 | the 15 seeds, same fields |
| `cases/research_question_mid.json` | designs 1–5 | the review's question, in scope and out of scope |
| `validation/wrong_question_for_negative_control.json` | design 5 | a *real* question from the adverse case, used as the wrong question |
| `cases/seed_selection.json` | design 2 | seed DOIs, keyed by case |

Why 311 and not everything: the intervals this size produces already separate 0.58 from 0.83
agreement, which is the distinction that matters. Scoring all 2,500 costs eight times more and
narrows nothing you would act on.

---

## Step 4 — score once (the only step that costs money in designs 1, 2, 3, 6)

    .venv/bin/python validation/llm_screen_validate.py score \
      --records validation/validation_records.json \
      --question cases/research_question_mid.json \
      --out cases/xvendor_scores.json

Roughly 112k input tokens per vendor — about 0.56M across five vendors, a few cents to a few dollars
depending on which models. Runs in a handful of minutes.

Watch the last two lines. `coverage by rater count` should show most records scored by every vendor.
If a vendor scored far fewer, its batches failed; the affected records are listed under `failures`
in the output file and are excluded from every statistic rather than silently averaged.

---

## Step 5 — the four free designs

No API calls. Run them in this order, because the first one determines whether the rest mean
anything.

**5a. Convergent validity against non-model signals (design 6).** Run this first. It is the only
check whose signal does not come from a language model.

    .venv/bin/python validation/llm_screen_validate.py bibliometric \
      --scores cases/xvendor_scores.json \
      --records validation/validation_records.json \
      --seed-records validation/validation_seeds.json \
      --out cases/xvendor_bibliometric.json

Read `topical_mean_auc` against `prestige_mean_auc`. Topical above prestige means the scores track
topic — good. **Prestige above topical means the scores track citation counts or venue standing, and
you should stop here**: no amount of further model calls fixes that, and no positive control would
have caught it.

**5b. Agreement, within versus between vendor families (design 1).**

    .venv/bin/python validation/llm_screen_validate.py agreement \
      --scores cases/xvendor_scores.json --out cases/xvendor_agreement.json

`between_family` is the number that answers a reviewer. `within_family` is the weaker claim the
manuscript currently makes. If `n_families` is 1, the script says so in `warning` and the run has
not achieved what you set out to do.

**5c. Blind positive controls (design 2).**

    .venv/bin/python validation/llm_screen_validate.py controls \
      --scores cases/xvendor_scores.json \
      --seeds cases/seed_selection.json --case mid \
      --out cases/xvendor_controls.json

Check `n_seeds` in the output. It should be close to 15; if it is much lower, some seeds were not
scored by every vendor and the common-basis guard excluded them. Read `per_seed` before drawing any
conclusion — a low-scoring seed can be a genuinely off-topic seed admitted by the Boolean query
rather than a screener failure.

**5d. Threshold sensitivity (design 3).**

    .venv/bin/python validation/llm_screen_validate.py threshold \
      --scores cases/xvendor_scores.json --out cases/xvendor_threshold.json

Report the whole band in the paper, not τ = 7 alone. `boundary_share` tells you what fraction of
records sit in the 6–8 zone where the cut decides the outcome.

Shortcut for 5b–5d in one pass:

    .venv/bin/python validation/llm_screen_validate.py validate \
      --scores cases/xvendor_scores.json \
      --seeds cases/seed_selection.json --case mid \
      --out cases/xvendor_validation.json

---

## Step 6 — the two designs that score again

These change the prompt, so they cannot reuse step 4's scores. Use `--limit 150` to keep the cost
proportionate; 150 records was enough to detect a 21% shift in the included set.

**6a. Wording and order robustness (design 4)** — three scoring passes.

    .venv/bin/python validation/llm_screen_validate.py robustness \
      --records validation/validation_records.json \
      --question cases/research_question_mid.json \
      --limit 150 --out cases/xvendor_robustness.json

Look at `conditions.paraphrase.tau_7.0`: `included_shipped` against `included_variant` is how much
the included set moves when the rubric is reworded but the criteria are not, and
`flips_in_boundary_zone` tells you whether the movement is confined to borderline records (it was,
13 of 14, in the single-model run already done).

**6b. Negative control with a wrong question (design 5)** — two scoring passes.

    .venv/bin/python validation/llm_screen_validate.py negative \
      --records validation/validation_records.json \
      --question cases/research_question_mid.json \
      --wrong-question validation/wrong_question_for_negative_control.json \
      --limit 150 --out cases/xvendor_negative.json

`include_collapse` should be large and `pearson_between_questions` low. A small collapse means the
screener is rewarding something other than topical relevance.

---

## Step 7 — the human sample, and why the manuscript does not use it

**Not part of the submission.** The tooling is here because it is useful for your own reviews, but
the manuscript reports no human-agreement coefficient, deliberately.

The reason is that the only available screener is the author, who defined the research question,
selected the seeds and wrote the rubric the models were given. An author-versus-ensemble κ measures
how faithfully the models reproduce the framing they were handed — a high value is as consistent
with a well-specified prompt as with a well-judging screener, and a low one is as consistent with a
badly-worded rubric. Neither reading supports a claim about screening quality, which is why
screening guidance requires two *independent* reviewers. Reporting the number anyway would invite a
reviewer to read it as expert validation, which it is not, and the paper's claim is about a library
that does not screen at all.

What the manuscript reports instead: blind seed controls (the ensemble separates known-relevant
records from the pool), cross-family agreement across six vendors (the scores are not one lineage's
artefact), and non-model bibliometric convergence (the scores track topic, not prestige).

If you have a colleague who did *not* write the protocol, the comparison becomes meaningful and the
tooling is ready — about three hours of their time:

1. Give them `llm_screening/human_validation_sample.csv` — 200 records, stratified across the score
   range, shuffled, **with the model scores withheld** in a separate key file. That separation is
   what makes the comparison blind; they must not see the key file.
2. They fill the `decision` column with `include` or `exclude`, judging against the research question
   in `cases/research_question_mid.json`.
3. Score it:

       .venv/bin/python validation/score_human_agreement.py

   It reports Cohen's κ with a confidence interval, PABAK, the confusion matrix, and the ten
   disagreements with the highest model scores — the same statistics ReviQ reports, so the
   comparison is like for like.

**If you do run it, decide how you will report the number before you see it.** High κ validates the
included set; low κ is the more interesting result, since it would show the ensemble is unsuitable
as a sole screener and thereby vindicate the library's refusal to screen. Choosing what to report
after seeing the value is the one indefensible option.

---

## What to put in the paper

| Ran | Goes in | Sentence it earns |
|---|---|---|
| Design 6 | Section S9 (replace the current figures) | the screener tracks topic, not prestige |
| Design 1 | Section S2 and the limitation in Impact | drop "within one model family" from the limitations if between-family agreement is good |
| Design 2 | Section S2 | blind controls separate, on a stated common basis |
| Design 3 | Section S8 | a reported sensitivity band instead of a single τ |
| Design 4 | Section S8 (replace the single-model figures) | the ranking is stable; the cut is what moves |
| Design 5 | Section S2, new paragraph | the screener responds to the question |
| Step 7 | *not in the submission* | only meaningful with a screener who did not author the protocol |

Designs 1 and 6 plus step 7 carry almost all the value. If time is short, run those three.
