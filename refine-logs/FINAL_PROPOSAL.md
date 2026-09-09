# Frozen proposal: same-event reanchor onset-to-rollout audit

## Problem anchor

Test whether a label-free local-to-remote transition at query `q` contributes to
the first hallucinated target `q+1`, then propagates through generated local states
to later hallucinated targets before another reanchor occurs. Do not infer this
trajectory by combining separate onset and all-H aggregate plots.

## Frozen estimands

1. Freeze reanchor coordinates with the existing source-max discovery before
   labels are joined.
2. Define target-specific `logit(correct)-logit(error)` contrasts independently of
   event selection, covering onset and subsequent span targets.
3. Propagate the discovery-aligned adjacent-row remote attention delta as the
   primary seed. Retain the whole current native remote write as a separate
   reference estimand.
4. Retain exact `0`, `1`, and `2+` position-hop effects. The candidate temporal
   chain requires an onset at offset 1 and continuing-H `2+`-hop effects before the
   next frozen event, all from the same event.
5. Use four disjoint provenance groups only for additive accounting. Propagate
   every annotated `source_unit_id` separately and report opposing/cancelling
   effects.
6. Require pointwise closure independently for margin, layer trajectories, and
   root coefficients. Hash all upstream artifacts and source annotations into
   resume identity.
7. Report normal, onset, and continuing targets symmetrically. Never turn a coarse
   source bucket into a named hallucination mechanism.

## Claim boundary

The implemented audit localizes an exploratory first-order candidate route on a
fixed trajectory. It does not establish binding, finite necessity/sufficiency,
free-generation change, or a general hallucination mechanism. Confirmation needs
matched normal pseudo-onsets, bidirectional exact route interventions with sham
controls, local-carrier mediation, paired free-run decoding, and held-out
cross-model/task replication.
