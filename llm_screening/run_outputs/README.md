# Run outputs, preserved for audit

`../../run_llm/` is a 165 MB working directory (state, 472-entry response cache, per-iteration
exports) and is excluded from version control by the `run_*/` rule. These are its small,
citable products:

- `prisma.json` -- PRISMA 2020 counts, per iteration, direction and provider
- `prisma.svg` -- the flow diagram
- `included.csv` -- the included studies
- `report.md` -- the library's own run report
- `run_manifest.json` -- resolved config plus artefact hashes (`verify` checks against this)

The full run directory reproduces from its cache offline. To hand it to a reviewer, archive
`run_llm/` itself; these files are what the manuscript cites.
