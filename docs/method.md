# Method specification

## Research object

The research object is a **Reanchor Transition**, not an attention node. For a
response query position `q`, a transition is evidence that attention changed
from recent ordinary context toward non-local ordinary sources, after correcting
for the number and dependence of layer/head tests. Consecutive significant
positions form one **Reanchor Episode**. Its strongest position is the
**Reanchor Anchor** traced by the causal stage.

This vocabulary prevents three distinct objects from being conflated:

- a **read site** is one `(layer, head, query)` measurement;
- a **transition** is one statistically calibrated query position;
- an **episode** is a consecutive run of transitions represented by one anchor.

## Stage 0: fixed-trajectory capture

Input is the complete `prompt + generated response` token sequence. A causal
teacher-forced forward pass records values already computed by the model:

- post-projection, post-RoPE queries and keys;
- projected values and per-head attention outputs;
- residual, attention and MLP writes at response queries;
- final residuals and observed-versus-runner readout directions;
- token, special-token, prompt-evidence and source-unit metadata.

Q/K/V arrays are activations, not checkpoint parameter matrices. Checkpoint
weights remain separate and are loaded lazily for causal tracing.

The compatibility adapter accepts `attention_audit_v3` without rewriting it.

## Stage 1: label-free transition features

For every ordinary adjacent response query pair `(q-1, q)`, reconstruct complete
native attention. Special tokens remain in the model computation but are
excluded from the event estimand. Let `w` be the local window and use the
current query partition for both rows:

```text
remote(q, s) := q - s > w
local(q, s)  := q - s <= w
```

Using the current partition for the previous row prevents a stationary source
from becoming an event merely because it aged across the window boundary.

Each read site records:

- previous and current local/remote mass;
- signed remote gain and total positive remote gain;
- total-variation change;
- focality, effective source count and gain-weighted distance;
- peak source and prompt/evidence/response-history gain fractions.

No correctness or hallucination label is available in this stage.

## Stage 2: independent-source max-null calibration

Raw thresholds are not event definitions. With `L x H` read sites at every
token, an `any(layer, head)` rule makes token selection approach one as model
size grows.

Two predeclared token statistics are computed before calibration:

```text
sparse(q) := maximum gated remote gain over all layers and heads
broad(q)  := maximum layer-wise mean of the strongest head fraction
```

The sparse channel detects an exceptional specialized head. The broad channel
detects a coordinated layer even when no single head is extreme. Because both
statistics already include their layer/head maximization, their null
distributions include the same multiplicity.

Within each `(task, relative-position bin, channel)`, every independent
`source_id` contributes its maximum score over all of its samples and tokens.
This makes the calibration unit an independent source rather than a correlated
token or head. The one-sided empirical max-null p-value uses

```text
p = (1 + count(source maxima >= observed)) / (n_sources + 1).
```

If a position stratum is too small, calibration falls back to a task/channel
source maximum and records that fallback. It never uses N/H labels.

## Stage 3: token selection and episodes

Bonferroni correction covers the two score channels and all predeclared
relative-position bins. The resulting family p-value controls the probability
that a null independent source produces any selected token under exchangeability
with the calibration sources. This is deliberately more conservative than
per-token FDR.

An eligible token is significant only if it satisfies all frozen requirements:

- corrected family `p <= alpha`;
- a positive remote-gain effect-size floor;
- the previous-local precondition;
- at least one active sparse or broad read-site pattern.

Consecutive significant positions are one episode. The position with the
smallest corrected family p-value, then largest remote gain, is its anchor. Only anchors
enter expensive causal tracing; all episode members remain in the artifact.

## Stage 4: morphology

The anchor is described rather than reselected. The dominant layer is the layer
with the largest supported positive gain. Frozen descriptors distinguish:

- broad versus sparse head participation;
- focal versus diffuse per-head gain;
- convergent versus diverse peak-source agreement;
- prompt, evidence and response-history source fractions;
- effective source count and mean retrieval distance.

The five initial morphology names are `broad_diffuse`, `broad_convergent`,
`broad_diverse`, `sparse_focal`, and `sparse_diffuse`.

## Stage 5: analytic causal tracing

For each frozen anchor read site, the seed is its native non-local attention
message, not its scalar attention gain:

```text
delta x = W_O^h sum_remote A[h,q,s] V[h,s].
```

Analytic Jacobian-vector products propagate this seed through native RMSNorm,
attention, residual and SwiGLU operations at the captured trajectory. Results
retain:

- `full`, `fixed_qk`, and `no_mlp_paths` variants;
- exact `0`, `1`, and `2+` position-hop accounting;
- observed-versus-runner or explicitly supplied candidate-margin responses;
- signed V-content and K-routing last-crossing edges.

Cut closure must reconstruct the complete one-or-more-hop local response before
an edge artifact is committed.

## Stage 6: outcome reporting

Only after anchors and edges are frozen are token labels joined. Reports keep
separate:

1. event incidence among all eligible tokens;
2. morphology-conditional future outcomes;
3. causal candidate-margin response conditional on a traced event;
4. coverage of unscored tokens and incomplete captures.

Token observations are summarized within an independent source first. Confidence
intervals resample sources, not correlated tokens, heads or events.

Morphology-conditioned reports separate the anchor target from later horizons.
Signed candidate-margin responses are also retained after every layer. A
positive intermediate explicit-candidate effect followed by a nonpositive final
effect is reported as `received_then_overridden`. This name is used only for an
externally defined positive/negative candidate contrast. Under the default
observed-versus-runner contrast, the same number is only a preference reversal:
the observed token may itself be hallucinated.

## Interpretation limits

- Capture is teacher-forced analysis of a fixed trajectory, not a new free run.
- Attention redistribution is a discovery signal, not proof of information use.
- JVP tracing is a local first-order causal sensitivity, not a finite ablation.
- Observed-versus-runner margins are not truth contrasts. Truth claims require
  explicit, independently defined correct/incorrect candidates.
- The source-max family correction controls the declared selection family under
  source exchangeability; it does not remove dataset shift, label error or
  causal-identification assumptions.
