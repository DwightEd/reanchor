# Input data contract

## What `attention_audit_v3` is

The input is a teacher-forced analysis of a fixed generation trajectory. A
response has already been generated. The capture then feeds the complete
`prompt + response` token sequence through the causal model once and records
the computations at response-query rows. The causal mask still prevents every
query from seeing future tokens.

This is not a second sampled generation, and it is not a collection of model
parameters. It is a collection of activations and metadata computed on the
observed trajectory.

## Files for one sample

For a compact path such as `train/QA/11859.npz`, companions use the same stem:

| File | Main content | Used by |
|---|---|---|
| `11859.npz` | token IDs/text, response rows, masks, source units, summarized native attention and write accounting | discovery metadata, reporting |
| `11859.qk.npz` | post-projection, post-RoPE Q/K activations; original compute dtype and attention scale | exact full-source attention reconstruction |
| `11859.history.npz` | native response-to-response attention probabilities for every layer/head/query | restores the exact captured history block |
| `11859.states.npz` | layer-input residuals, projected V activations, head codes, attention/MLP writes and final residual | analytic causal tracing |
| `11859.labels.npz` | response-token labels `-1/0/1` | reporting only |
| `11859.attention.npz` | optional full dense attention rows | diagnostics only; not required |

Q, K and V here are computed activations. They are outputs of the checkpoint's
projection layers (Q/K are also after RoPE), not `q_proj`, `k_proj` or `v_proj`
weight matrices. The tracing stage loads those matrices separately and lazily
from the checkpoint identified by `index.json`.

`row_position` normally spans `response_start - 1` through the last input token.
The first row predicts the first response token. The final row has no captured
next-token outcome, so causal reports exclude it as a target.

## Why both Q/K and native history are stored

Q/K plus scale and the causal mask reconstruct the complete-source attention
row, including all prompt sources, without storing a very large dense tensor.
The reconstructed prompt block is a numerical replay from stored activations;
another device can differ slightly because of kernel and dtype arithmetic.
Therefore the response-history block is overwritten with probabilities from
the original forward pass. No source is imputed, thresholded or replaced with
a top-k approximation.

## Label firewall

The capture manifest must declare `labels_used_for_capture=false`. Discovery
opens only the compact, Q/K and history files. Tracing additionally opens
states and checkpoint weights. Only `ReportBuilder` opens `.labels.npz`.

The same `source_id` may supply multiple correlated samples inside one split,
but may not appear in both calibration and held-out splits.
