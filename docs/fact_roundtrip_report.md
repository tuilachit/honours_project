# FinancialFact round-trip report

## Result

The representation gate passes on the approved 500-question synthetic set.

| Check | Result |
|---|---:|
| Approved questions | 500 |
| Complete source tables | 500 |
| Numeric `FinancialFact` records materialised | 8,082 |
| Approved target cells recovered | 500 / 500 |
| Natural hard-negative cells recovered | 1,000 / 1,000 |
| Unique stable cell IDs | 8,082 / 8,082 |
| Unique versioned fact IDs | 8,082 / 8,082 |
| Exact serialize/deserialize matches | 8,082 / 8,082 |

## What this tests

For each source table, the workflow parses all numeric cells first. The parser
receives the table, dataset revision, source identity and configured coordinate
transform, but it does not receive the approved target or hard-negative cell
IDs. Only after the complete table has been converted are the 1,500 labelled
cells looked up by stable cell ID.

Each fact preserves:

- the original displayed value;
- its normalized decimal when unambiguous;
- source document, context, page, table, row and column provenance;
- raw row and multi-level column-header paths;
- entity and period context;
- generic header dimensions that do not require a fixed financial schema; and
- a versioned fact ID distinct from the stable source-cell ID.

The generated result is stamped with the Git commit, resolved configuration,
configuration hash, timestamp and pinned dataset-manifest hash.

## Reproduction

```sh
make fact-roundtrip
```

The result is written to
`results/facts/synthetic_financial_fact_roundtrip.json`. Generated results stay
gitignored and can be recreated from the version-controlled configuration,
approved review log and pinned source data.

## Limits

This is a representation and provenance test, not a retrieval result. It shows
that the approved cells survive table-wide conversion; it does not show that a
search model can find them. B0 and B1 retrieval runs remain the next engineering
gate.

The first-pass labels were reviewed by the primary researcher with Codex
suggestions visible. A blind independent-review sample is still required before
confirmatory evaluation. Currency, scale and percentage fields are conservative
deterministic interpretations that retain their raw markers; later retrieval
experiments must report normalization failures separately rather than treating
this round trip as proof of complete semantic parsing.
