# Financial Regulatory RAG Evaluation

Evaluation harness for the honours thesis **Evaluating and Reducing Hallucinations in RAG for Financial Regulatory Question Answering**.

## Experimental conditions

- **C1:** hybrid dense + BM25 retrieval, cross-encoder reranking, and LLM generation.
- **C2a:** C1 plus NLI and LLM-judge verification for every atomic claim.
- **C2b:** C1 plus type-routed verification: textual claims use NLI and an LLM judge; numeric and threshold claims use deterministic arithmetic.
- **C3:** C2b plus calibrated abstention based on an answer support score.

The headline comparison is verifier miss-rate by claim type on a hand-labelled gold set, comparing C2a with C2b.

## Repository layout

The `src/` packages separate dataset normalization, retrieval, generation, claim processing, verification, abstention, and evaluation. Condition YAML files in `configs/` are merged with `configs/base.yaml`. Entry points live in `scripts/`, generated artifacts in `results/`, and interface tests in `tests/`.

This initial scaffold defines interfaces only. Experiment modules intentionally raise `NotImplementedError`; configuration merging, stable hashing, and stamped JSON result writing are implemented so all later components share one provenance path.

## Environment

Python 3.11 and `uv` are required. Dependencies are pinned in `uv.lock`.

```sh
make setup
make test
```

Once the experiment modules are implemented, `make eval` is the single entry point for regenerating every condition result from a fresh clone, and `make figures` renders configured thesis figures.
