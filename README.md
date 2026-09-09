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

See [the method specification](docs/method.md) and
[the architecture](docs/architecture.md).

