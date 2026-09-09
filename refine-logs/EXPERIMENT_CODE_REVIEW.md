# Independent Method and Code Review

**Verdict: PASS — no blocking code findings.**

The implementation is ready to merge as a mechanism-audit instrument. It does
**not** yet establish an empirical hallucination mechanism; that requires the real
P002/confirmation runs.

## Objective fit

The patch now matches the corrected objective:

- Reuses statistically frozen, label-free reanchor sites.
- Audits the discovery-aligned adjacent-row remote-attention delta, rather than
  treating cross-example activation patching as proof.
- Keeps the entire current remote write as a separate reference estimand.
- Follows onset and later continuing-H targets within the same event.
- Retains signed source-unit effects instead of collapsing all evidence into four
  purported mechanism classes.

Relevant implementation: `tracing/jacobian.py`, `tracing/mechanism.py`, and
`reporting/evaluation.py`.

## What the patch proves

Given a fixed captured trajectory and explicit correct/error token contrast, it can
establish that:

1. The constructed transition seed exactly follows discovery's ordinary-normalized
   adjacent-row attention delta.
2. Coarse provenance components add back pointwise to the full transition JVP.
3. A specific source unit has a signed first-order association with the later
   correct-minus-error margin.
4. The effect travels through direct or `2+` position-hop paths on the same frozen
   event trajectory.
5. Outcome span labels were joined only during reporting.

Closure is independently and pointwise gated for margin, layer trajectory, and root
coefficient.

## What it cannot prove

The patch alone cannot prove:

- that the transition caused the sampled hallucination;
- finite necessity or sufficiency;
- constraint/content relational binding;
- that deleting or restoring the route changes free generation;
- that the signal distinguishes hallucinations after matched controls;
- cross-task or cross-model generality.

The implemented estimand is a local JVP attribution direction. It is not an exact
attention intervention because it compares attention rows from different query
positions and retains only the remote part. The documentation states this boundary.

## Integrity checks

- **Source grouping:** Four groups are bookkeeping partitions, while nonnegative
  `source_unit_id` values retain document-level effects.
- **Label leakage:** Reanchor discovery remains label-free. The audit declares that
  it uses explicit correctness contrasts while not loading outcome-token labels;
  labels enter in reporting.
- **Resume identity:** Includes discovery and trace manifests, transition/event
  artifacts, QK/history/states, source annotations, full trace, settings, and
  contrast digest. Frozen event sites, variants, hop names, and seed kind are
  checked.
- **Reporting taxonomy:** Unsupported causal labels were removed. Reporting exposes
  continuous quantities, target phase, and an explicitly exploratory same-event
  candidate chain.
- **Performance:** Source groups and units share layer construction and attention
  reconstruction in one partitioned propagation pass.

## Nonblocking follow-ups

- CPU state memory still scales with `events × partitions × rows × hidden`; add a
  memory estimator or partition chunk limit before very large source-unit runs.
- `semantic_roles_available` means the `source_kind` field exists, not that every
  role was annotated. Per-role coverage would be clearer.
- Correct/error contrast validity remains an external data-contract assumption;
  record its provenance in released experiment metadata.
- A mutable checkpoint at the same path is an explicit trust boundary.
- Trace schema was bumped to `analytic-trace@2`; existing v1 run directories require
  retracing or a fresh output directory.

## Verification

- `pytest -q`: **74 passed**, 2 warnings.
- Ruff: **all checks passed**.
- `git diff --check`: clean.
- Pytest printed the known Windows `pyarrow`/`pandas` import access-violation
  diagnostic after completion, but returned exit code 0; this appears
  environment-specific rather than a failing project test.
