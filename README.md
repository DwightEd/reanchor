# Reanchor

Reanchor studies token-level transitions in which a causal language model moves
from recent context toward non-local prompt or response information, and tests
whether the retrieved message changes later token preferences.

The project deliberately separates three claims:

1. **Capture:** what the model computed on a fixed generation trajectory.
2. **Discovery:** which token transitions survive label-free null calibration and
   multiple-testing correction.
3. **Tracing:** how a frozen event message locally propagates through the native
   Transformer computation graph.

Hallucination labels are outcome annotations. They never select events or graph
edges.

## Project layout

```text
src/reanchor/
|-- cli.py                 # parse arguments and invoke the pipeline
|-- pipeline.py            # visible stage orchestration
|-- capture/               # fixed-trajectory feature protocol and capture
|-- discovery/             # transition features, calibration, selection, morphology
|-- tracing/               # analytic JVP propagation and signed transport cuts
|-- reporting/             # label join, source-balanced estimates and rendering
`-- artifacts/             # versioned, atomic artifact persistence
tests/                     # public-seam and end-to-end tests
docs/
|-- method.md              # estimands, selection rules and interpretation limits
`-- architecture.md        # module interfaces, data flow and file responsibilities
```

The initial input adapter reads existing `attention_audit_v3` captures, so the
completed capture does not need to be regenerated during migration.

The mechanism-discovery work lives in
[`experiments/constraint_control`](experiments/constraint_control). Its first
vertical slice samples new answers autoregressively, replays the exact sampled
tokens, verifies logit fidelity, and stores label-free intermediate states. It
does not treat the old audit answers as free-run trajectories.

See [the method specification](docs/method.md) and
[the architecture](docs/architecture.md). The exact input arrays are documented
in [the data contract](docs/data.md).

The current free-generation P001 experiment reads the original RAGTruth
`source_info.jsonl` directly; no hand-written `data/questions.jsonl` is needed.
On the configured GPU server it can be launched with:

```bash
conda run --no-capture-output -n research \
  bash scripts/run_constraint_control_p001.sh
```

RAGTruth's existing responses and hallucination spans are not reused as labels
for the newly sampled answers.

## Install and run

```bash
git clone https://github.com/DwightEd/reanchor.git
cd reanchor
python -m pip install -e .

python -m reanchor run \
  --capture /path/to/attention_audit_v3 \
  --output outputs/reanchor_v1 \
  --device cuda:0
```

The portable one-command script accepts the capture root, output root and an
optional device:

```bash
bash scripts/run_v3.sh \
  /path/to/graph/experiments/reanchor_flow/outputs/attention_audit_v3 \
  outputs/reanchor_v1 \
  cuda:0
```

`discover`, `trace` and `report` can be run separately in that order. Completed
transition and trace artifacts are identity-checked and resumed. The output is
always separate from the immutable capture.

An optional candidate file makes correctness-oriented tracing explicit:

```json
{
  "test/QA/11859": [
    {"target": 123, "positive_id": 42, "negative_id": 91}
  ]
}
```

Pass it with `--contrasts contrasts.json`. `target` is an absolute predicted
token position. The file content hash becomes part of trace identity, so a
changed contrast cannot silently reuse an old trace.

To ask how the selected attention transition affected that candidate decision,
run the mechanism audit with the exact same contrast file:

```bash
python -m reanchor audit \
  --capture /path/to/attention_audit_v3 \
  --output outputs/reanchor_v1 \
  --contrasts contrasts.json \
  --device cuda:0

python -m reanchor report \
  --capture /path/to/attention_audit_v3 \
  --output outputs/reanchor_v1
```

This audit propagates the adjacent-row remote attention innovation that discovery
actually selected. It reports normal/onset/continuing targets symmetrically and
keeps exact `0`/`1`/`2+` position-hop effects. Four coarse source groups are only a
complete provenance partition, not four hallucination mechanisms; annotated
`source_unit_id` effects are retained separately. The prior current-remote-write
trace remains a reference estimand. Pointwise additive closure is required.

The audit is still a fixed-trajectory linearized candidate-route test. A causal or
general mechanism claim requires matched pseudo-onsets, bidirectional exact route
interventions with shams, free-run endpoints, and held-out cross-model/task
replication. Binding claims additionally require matched constraint
counterfactuals.
See [the mechanism-audit specification](docs/mechanism_audit.md).

## What gets selected

A raw read-site threshold is never called a reanchor event. The workflow first
forms two token statistics: a sparse exceptional-head score and a broad
coordinated-head score. Each independent training `source_id` then contributes
its maximum score across its correlated samples and tokens. Empirical
source-max p-values are corrected across the predeclared score channels and
relative-position bins. Consecutive significant tokens form an episode, and
only its strongest anchor is causally traced.

This directly addresses the former `any(layer, head)` trap, where the chance of
selecting a token grows toward one as the number of read sites grows.
