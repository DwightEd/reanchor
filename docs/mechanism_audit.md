# Reanchor-conditioned mechanism audit

## Claim being tested

The main hypothesis is temporal, not a four-way source taxonomy:

> At an `N -> H` onset, a label-free reanchor transition writes an
> error-favouring remote-source innovation into the response state; before a new
> reanchor occurs, that innovation reaches later hallucinated tokens mainly
> through `2+` generated-token position hops.

The current `2+` bucket means two or more position-to-position transfers; it does
not require every transfer to stay inside the configured local window. A local
carrier claim therefore remains untested until edge distances are partitioned or
the proposed local carriers are cut directly.

The audit must test both parts on the **same frozen event trajectory**. An average
over all hallucinated tokens and a separate average over onset tokens cannot be
joined into this sequence.

## Status of the legacy graph observation

The legacy `graph` repository records a real exploratory result from the supplied
`reanchor_evidence_20260908.zip`, but also states that the original 8B forward pass
was not rerun in that environment. For each cohort it counted 1,024 physical heads
passing BY-corrected tests:

| metric | all matched H: increase / decrease | annotated onset: increase / decrease |
|---|---:|---:|
| `local_history_share` | 998 / 7 | 34 / 366 |
| `far_history_share` | 151 / 634 | 646 / 9 |
| `distance_clip10` | 37 / 927 | 370 / 46 |
| `change_tv` | 7 / 928 | 192 / 18 |
| `prompt_share` | 87 / 700 | 80 / 349 |

There were 7,097 matched-H targets from 303 independent sources in the top cohort
and 259 onset events from 189 sources in the onset cohort. Heads are correlated and
are not independent replications. `far_history_share` means distant **earlier
response** tokens, not prompt constraints; prompt share more often decreased even
at onset. The repository therefore calls these numbers structural association, not
a causal trajectory. See the committed
[`LOOKBACK_EVENTS.md`](https://github.com/DwightEd/graph/blob/main/experiments/reanchor_flow/message_dag/LOOKBACK_EVENTS.md#已验证结果与真实数据覆盖).

## Why four source groups still appear

`constraint`, `content`, `other_prompt`, and `response_history` are a disjoint
provenance partition used to check additive accounting. They are declared before
outcomes are joined; they are not four discovered causes or four hidden states.

- `constraint`: typed question/scope/time/negation/relation constraints;
- `content`: source passages or structured evidence;
- `other_prompt`: other ordinary prompt tokens;
- `response_history`: already generated ordinary tokens outside the local window.

If token-level `source_kind` is absent, `evidence_mask` becomes `content`,
`constraint` stays empty, and the artifact says semantic roles are unavailable.
Every nonnegative `source_unit_id` is also propagated separately. This prevents two
documents with opposite signed effects from cancelling inside `content`. The four
coarse groups remain secondary interpretation coordinates.

## Estimands implemented by `audit`

Discovery selects an adjacent-row attention redistribution. For current query `q`,
the same current legal-source set and ordinary-token normalization are used for
both rows:

```text
I_q^h(s) = [Abar_q^h(s) - Abar_(q-1)^h(s)] 1[q-s>w]
delta_x_transition = W_O^h sum_s I_q^h(s) V_q^h(s)
```

The primary audit propagates `delta_x_transition` through the native analytic JVP.
Without a candidate file the target is
`M_t = logit(generated_t) - logit(runner-up_t)`. With an independently supplied
contrast it is `M_t = logit(correct_t) - logit(error_t)` at the declared target.
The two readouts are reported separately. This aligns the mechanism estimand with
the event that discovery selected, but remains an attribution of the attention
innovation rather than an exact finite intervention.

The previous trace's current native remote write is retained separately:

```text
delta_x_current = W_O^h sum_remote A_q^h(s) V_q^h(s)
```

It answers what the whole current remote write locally contributes and must not be
named “the effect of the transition.” Reports expose both quantities and exact
`0`, `1`, and `2+` position-hop components.

All coarse groups and all annotated source units are propagated in one shared
layer/attention pass. Margin, per-layer, and root-coefficient closure are checked
independently and pointwise:

```text
abs(sum_group component - full_transition)
    <= closure_atol + closure_rtol * abs(full_transition)
```

A large effect in another field or target cannot hide a local closure failure.
Artifact resume identity includes the discovery manifest, trace manifest,
transition/event artifacts, QK/history/state captures, source annotations, full
trace, settings, and contrast content hash. The checkpoint path/revision recorded
by the immutable capture manifest is a trust boundary; a mutable checkpoint reused
at the same unresolved path is not supported.

## Same-event temporal ledger

Labels are joined only by `report`. Every target is marked as:

- `normal`: label `N`;
- `onset`: label `H` with an ordinary preceding `N` token;
- `continuing`: label `H` with a preceding `H` token;
- `hallucinated_boundary_unknown`: the preceding outcome is unavailable.

Query `q` predicts target `q+1`, so the primary onset alignment is exactly
`event_target_offset=1`. A later `continuing` target is a rollout target only when
it precedes the next frozen reanchor. This avoids attributing a later error to an
earlier event after a new remote update has occurred.

`reports/mechanisms.csv` contains the readout kind, target phase, transition seed
norm, transition/current margin effects, effect per seed norm, hop effects, group
cancellation, and all closure diagnostics.
`reports/mechanism_source_units.csv` contains signed effects and absolute-effect
ranks for every annotated source unit. `summary.json` reports normal, onset, and
continuing phases symmetrically for each readout and includes a same-event
onset-to-rollout negative-margin rate. Under observed-versus-runner this means the
route disfavors the emitted token; only an explicit correct-minus-error readout can
call a negative effect error-favouring. The rate is marked
`exploratory_linearized_candidate_only` and `claim_supported=false`. The audit does
not open N/H outcome-token labels; explicit contrast files remain optional external
candidate supervision.

## Confirmation experiment required for a general mechanism claim

Freeze this protocol before looking at the confirmation split.

1. **Independent units and scope.** Split by `source_id`, never token/head. Use at
   least two model families and two task families, with per-domain estimates and a
   hierarchical cross-domain effect. Report samples with no event, missing
   candidate truth, failed replay, and failed intervention.
2. **Matched onset association.** Compare every `N -> H` onset with two-sided
   all-normal pseudo-onsets from the same sample/source, matched on response
   position, coarse token type, log probability, entropy, and available context.
   Use source-cluster permutation/max-T simultaneous intervals over the complete
   predeclared `[-K,+K]` offset curve. Do not count significant heads as samples.
3. **Same-event transition test.** At `q -> q+1`, require an explicit candidate
   contrast at the onset and at each later labelled token. Estimate whether the
   signed transition effect is more error-favouring at onset than at matched normal
   controls. Then test whether its `2+`-hop effect remains error-favouring on
   continuing-H targets before the next reanchor. Report the conjunction and every
   denominator, not two separate aggregate means.
4. **Exact necessity and sufficiency.** Rerun the model after deleting the selected
   value-path write without attention renormalization, and in the reverse direction
   restore the matched correct-source write. Use same-distance, random-head,
   random-layer, and magnitude-matched sham routes. A proposed trigger needs a
   larger correct-minus-error margin improvement than shams in both directions.
5. **Local-carrier mediation.** After onset, cut only the identified local carrier
   edges or patch the pre-onset clean residual while leaving the prompt and prefix
   fixed. The later hallucinated-span probability/length should fall, and the remote
   intervention effect should attenuate when the local carrier is already cut.
6. **Free-run endpoint.** Teacher-forced margins localize a route; they do not show
   that generation changes. Branch decode from the intervention point with paired
   seeds/shared random numbers and score factual correctness, hallucinated-span
   length, and semantic answer changes.
7. **Predeclared falsifiers.** Reject the mechanism if the onset curve disappears
   after matching, the selected remote source is corrective rather than harmful,
   exact deletion is no stronger than shams, the later `2+`-hop/local cut has no
   effect, or the sign fails leave-one-model/task-out replication.

Passing only the observational or JVP stages supports “candidate route” language.
Passing the bidirectional exact interventions and free-run endpoint supports a
causal mechanism within the tested model/task scope. Even replicated success should
be called a recurrent mechanism family, not a universal mechanism of all LLM
hallucinations.

## Running after a completed trace

The completed trace and audit must use the same contrast setting. With no contrast
file, both stages use generated-versus-runner readouts. For correctness-oriented
rollout, the file must cover the onset target and later labelled span targets.

```bash
python -m reanchor trace \
  --capture /path/to/attention_audit_v3 \
  --output outputs/reanchor_v1 \
  --contrasts contrasts.json \
  --device cuda:0

python -m reanchor audit \
  --capture /path/to/attention_audit_v3 \
  --output outputs/reanchor_v1 \
  --contrasts contrasts.json \
  --device cuda:0

python -m reanchor report \
  --capture /path/to/attention_audit_v3 \
  --output outputs/reanchor_v1
```
