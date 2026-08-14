# Independent re-check of the three-case results

Everything below was recomputed from the saved artifacts, not taken from the sub-agents' reports.

## Reproduces exactly

- **PRISMA arithmetic.** screened = excluded + included in all three cases
  (1551 = 1402 + 149; 2500 = 2163 + 337; 3000 = 2314 + 686).
- **Arm capture cells.** Sum to the stated observed count in all three (164, 352, 701), each of
  which is the included count plus the 15 seeds.
- **Label files against PRISMA.** Include counts match exactly in all three.
- **ICC(2,1).** Recomputed from the raw score matrices: 0.578 / 0.827 / 0.795, matching the
  reported figures to three decimals.
- **Blind control AUC, favourable and large cases.** 0.658 and 0.631, exact.

## Does not reproduce

**The adverse case's control AUC of 0.781 could not be recovered.** Recomputing from
`cases/scores_adverse.json` gives 0.580 on all scored records, 0.614 using the first-stage score as
a common basis for everything, and 0.733 restricted to gate survivors. The cause is visible in the
artifact's own per-seed table: two of the fifteen seeds were gated out at stage one and carry a
single model's score, while survivors carry three, so the reported figure mixes bases. The exact
iteration-1 candidate pool cannot be reconstructed either, because `run_adverse/candidates.json`
was overwritten by iteration 2 (it now holds 31,486 keys).

Consequence for the manuscript: the adverse case's AUC should be reported as a range across bases,
or recomputed on a common basis, not as 0.781. The favourable and large figures are unaffected. The
new `validation/llm_screen_validate.py` enforces a common basis for exactly this reason.

**The adverse case's gate false-negative rate was never measured.** The artifact says so plainly:
350 records were sampled and their batches built, but the screening resource ran out before any
were scored. It is reported as "not measured", which is correct, and it means the gate's
false-negative rate rests on the favourable case's 0% alone.
