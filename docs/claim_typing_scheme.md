# Claim-Typing Scheme v0.1

**Project:** Evaluating and Reducing Hallucinations in RAG for Financial Regulatory QA
**Milestone:** 1 (Week 2)
**Status:** DRAFT — pending supervisor review (A/Prof Wei Liu)
**Purpose:** Defines how each atomic claim extracted from a generated answer is classified as TEXTUAL or NUMERIC/THRESHOLD, determining which verifier checks it.

---

## 1. Why this document exists

Claim typing is the routing decision at the centre of the project's contribution. Textual claims are checked by NLI + LLM-judge; numeric/threshold claims are checked by a deterministic arithmetic/constraint checker. A misrouted claim is a measured failure mode (§7), not an implementation detail: a numeric claim sent down the entailment path is graded by exactly the mechanism the project argues is structurally blind to it.

This scheme must be fixed **before** gold-set annotation begins, so that inter-annotator agreement (target Cohen's κ ≥ 0.7) measures genuine ambiguity in the task rather than drift in the rules. Any change after annotation starts is versioned, dated, and requires re-labelling of affected claims.

## 2. Scope: two classes only

The taxonomy is deliberately binary. FinGround uses a richer financial claim taxonomy; this project uses two classes because (a) the headline result only requires separating claims the arithmetic checker can handle from those it cannot, and (b) a two-class scheme is achievable at honours-scale annotation quality. The Week 7 gate already specifies fallback to a plain two-class split if a finer taxonomy underperforms — this scheme starts at that fallback by design.

## 3. The operating rule

> **A claim is NUMERIC/THRESHOLD if and only if its truth can be decided by extracting one or more values from the evidence and applying a comparator (equality, inequality, ordering, or arithmetic relation). Otherwise it is TEXTUAL.**

Every ruling below follows from this rule. When a new edge case arises during annotation, resolve it by asking: *can a deterministic checker decide this by pulling values and comparing them?* If the answer requires interpreting what a regulation means, it is TEXTUAL.

Two corollaries used throughout:

- **The presence of a number does not make a claim numeric.** "Section 4.5 of the standard requires prompt disclosure" contains a number that is an identifier, not a quantity.
- **The absence of a number does not make a claim textual.** "The ratio exceeded the regulatory minimum" is a comparison between two extractable values, even though neither appears in the claim text.

## 4. Class definitions

### 4.1 NUMERIC/THRESHOLD

A claim asserting a quantity, a relation between quantities, or a temporal point that can be checked against values in the evidence.

Includes:
- Reported values: "CET1 ratio was 12.9%"
- Regulatory thresholds: "the minimum requirement is 10.5%"
- Comparisons: "the ratio exceeded the minimum"
- Derived quantities: "revenue grew 8% year on year" (reconstructible from constituent values)
- Absolute amounts with units and scale: "net income of $4.2 billion"
- Dates and deadlines functioning as thresholds: "must be reported within 30 days"
- Counts: "the firm operates in 14 jurisdictions"

### 4.2 TEXTUAL

A claim asserting a definition, obligation, qualitative interpretation, causal relation, or scope of application, whose truth depends on the meaning of the evidence rather than on values extracted from it.

Includes:
- Obligations: "ADIs must notify the regulator of material breaches"
- Definitions: "Tier 1 capital comprises common equity and additional Tier 1 instruments"
- Scope: "the requirement applies to authorised deposit-taking institutions"
- Qualitative interpretation: "the standard permits a phased transition"
- Causal claims: "the increase was driven by retained earnings"
- Identifiers and references: "this is set out in Prudential Standard APS 110"

## 5. Edge-case rulings

Each ruling states the pattern, the decision, and the reasoning. Rulings marked **[REVIEW]** are the ones on which supervisor input is specifically sought.

### R1. Dates and deadlines → NUMERIC/THRESHOLD

"Must be reported within 30 days"; "the reporting period ended 30 June 2024".

Dates behave as thresholds: they are extractable values compared under an ordering relation. A deterministic checker can verify "within 30 days" or "as at 30 June 2024" against evidence without interpreting regulatory meaning.

*Boundary:* a date used purely as a label ("the FY2024 report") is not itself the claim's assertion — see R6 on decomposition.

### R2. Vague quantifiers → TEXTUAL

"Most ADIs are affected"; "substantially all subsidiaries"; "a material amount".

No comparator can be applied without first fixing an interpretation of "most" or "material". These are semantic judgments, and the arithmetic checker has no defensible policy for them. Routing them to the entailment judge is correct, not a compromise.

*Note:* "material" in a regulatory context is sometimes formally defined with a numeric threshold in the source document. If the claim explicitly invokes that definition ("exceeded the materiality threshold of $1M"), it becomes NUMERIC under R3.

### R3. Comparison without stated values → NUMERIC/THRESHOLD

"The ratio exceeded the regulatory minimum"; "revenue was higher than the prior year".

Both operands are extractable from evidence and the comparator is explicit. This is precisely the case the arithmetic checker exists to handle, and precisely the case an entailment judge handles badly — a passage discussing both quantities looks semantically supportive regardless of which is larger.

### R4. Identifiers, citations, and section numbers → TEXTUAL

"Set out in Section 4.5"; "under Article 26(1)"; "Prudential Standard APS 110".

The number is a name, not a quantity. No comparator applies. Attempting arithmetic verification here is a misroute in the opposite direction and would show up as a false failure.

### R5. Percentages as qualitative characterisations → TEXTUAL **[REVIEW]**

"A significant proportion of firms"; "the vast majority of filings".

Treated as vague quantifiers under R2. Flagged for review because the boundary with R3 is thin: if annotation shows annotators frequently reading these as implicit comparisons, the ruling may need tightening.

### R6. Mixed claims → decomposed further before typing

"The ratio increased to 12.4% due to retained earnings" contains a numeric assertion (the value, and the direction of change) and a causal assertion (the driver).

**Ruling:** this indicates incomplete decomposition, not a typing problem. Such claims are split at the decomposition step into "the ratio was 12.4%" (NUMERIC), "the ratio increased" (NUMERIC, comparison against prior period), and "the increase was driven by retained earnings" (TEXTUAL).

*Consequence for the pipeline:* the decomposer prompt must be instructed to split conjoined numeric-and-causal statements. If a mixed claim reaches the type classifier intact, it is logged as a decomposition failure, and typed by its **primary assertion** — the quantity — with the fallback recorded.

### R7. Units, scales, and currency → NUMERIC/THRESHOLD, with parse policy

"Net income of $4.2 billion"; "(2,341)" as a negative; "$1,234.5M".

Typed as numeric. Whether the checker can *parse* them is a separate question governed by the numeric-claim scope policy: unparseable values are logged and routed to abstention rather than to the entailment judge (per the proposal risk register). Typing and parseability are deliberately separate decisions — conflating them would let parser limitations silently reshape the claim-type distribution and contaminate the headline miss-rate.

### R8. Existence and negative claims → depends on the assertion

"The filing does not disclose a CET1 ratio" → TEXTUAL (an assertion about the evidence, not a value comparison).
"The CET1 ratio was not below the minimum" → NUMERIC (a comparator with negation).

### R9. Claims about regulatory applicability with numeric conditions → TEXTUAL **[REVIEW]**

"The requirement applies to institutions with assets above $50 billion."

Contains an extractable threshold, but the claim's assertion is about scope of application, not about a value. Provisionally TEXTUAL. Flagged for review: an argument exists for treating the embedded threshold as separately verifiable via decomposition under R6, and the ruling may change if such claims prove frequent in the pilot.

## 6. Annotation procedure for typing

1. Read the atomic claim in isolation, without the source answer.
2. Apply the operating rule (§3).
3. If uncertain, check §5 for a governing ruling.
4. If no ruling governs, assign a type, mark the claim `AMBIGUOUS`, and record the reasoning. Ambiguous cases are reviewed in batch and become new rulings in the next scheme version.
5. Never consult the verifier output when typing — typing precedes verification and must not be contaminated by it.

## 7. Misrouting as a measured outcome

Because typing determines routing, typing errors have downstream cost. Two directions are tracked separately:

- **Numeric → textual (severe).** A quantitative claim graded by an entailment judge — the exact blind spot the project critiques. Expected to produce silent misses.
- **Textual → numeric (recoverable).** A qualitative claim sent to the arithmetic checker, which will typically fail to parse it and route it to abstention. Costs coverage, not correctness.

Both are reported in the ablation on claim-typing accuracy (proposal §5.4a), including a sweep in which routing accuracy is synthetically degraded to measure how much of the type-routed advantage survives an imperfect router.

## 8. Version history

| Version | Date | Change |
|---|---|---|
| v0.1 | Week 2 | Initial draft for supervisor review |

**Open questions for supervisor:**
1. R5 and R9 rulings — agree, or tighten?
2. Should the embedded-threshold case (R9) be handled by forcing decomposition instead?
3. Is a two-class scheme the right starting point, or should sub-types be recorded as metadata now to allow finer analysis later without re-labelling?
