# Architecture

## Execution path

The default path stays linear:

```text
parse arguments
  -> construct PipelineConfig
  -> ReanchorPipeline.run()
  -> validate capture
  -> extract transition features
  -> calibrate and freeze episodes
  -> trace episode anchors
  -> join labels and report
```

`cli.py` owns only external parameters and presentation. `pipeline.py` owns stage
ordering and resume rules; it contains no attention mathematics.

## Modules and interfaces

| Module | Responsibility | Main interface |
|---|---|---|
| `capture.protocol` | Validate and read versioned v3 capture artifacts | `AuditDataset(root)` |
| `capture.attention` | Reconstruct complete attention rows from Q/K plus native history | `AttentionReader.rows(sample)` |
| `discovery.features` | Produce label-free read-site transition measurements | `TransitionExtractor.run(sample)` |
| `discovery.scores` | Collapse layer/head features into sparse and broad token statistics | `score_transitions(...)` |
| `discovery.calibration` | Independent-source max null and episode selection | `MaxNullCalibrator.fit(...).select(...)` |
| `discovery.events` | Collapse significant transitions into episodes and anchors | `EventDiscovery.run(dataset)` |
| `discovery.morphology` | Describe frozen anchors without reselection | `MorphologyProfiler.run(...)` |
| `tracing.tracer` | Load frozen anchors and orchestrate one sample at a time | `CausalTracer.run(...)` |
| `tracing.jacobian` | Native RMSNorm, attention and SwiGLU JVP/VJP operators | `DifferentialLayer` |
| `tracing.propagation` | Propagate messages with exact position-hop accounting | `trace_events(...)` |
| `tracing.cuts` | Persist and certify signed last-crossing transport edges | `CutRecorder` |
| `reporting.evaluation` | Join labels, future outcomes and source-balanced estimates | `ReportBuilder.run(...)` |
| `artifacts.store` | Atomic, versioned output persistence and resume identity | `ArtifactStore` |

These are package-internal modules. The supported user interface is the CLI and
the three workflow classes `EventDiscovery`, `CausalTracer`, and `ReportBuilder`.

## Dependency direction

```text
cli -> pipeline
pipeline -> capture, discovery, tracing, reporting, artifacts
discovery -> capture, artifacts
tracing -> capture, artifacts
reporting -> capture, artifacts
capture -> (no project package dependency)
artifacts -> (no project package dependency)
```

No lower module imports the CLI or pipeline. Discovery cannot import labels or
reporting. This dependency rule is the label-leakage firewall.

## Artifact ownership

The input capture is immutable. A run writes to a distinct output root:

```text
run/
|-- index.json
|-- calibration.json
|-- samples/<split>/<task>/<sample>/
|   |-- transitions.npz
|   |-- events.npz
|   |-- local_readout.npz
|   |-- traces/event_<position>.npz
|   `-- edges/event_<position>.npz
`-- reports/
    |-- summary.json
    `-- events.csv
```

Every decision and trace artifact includes schema, sample identity and frozen
method settings. Streamed edge artifacts include their cut schema and event
coordinates. Temporary files are committed by atomic rename only after
validation.

## Testing seams

- Pure feature extraction is tested from attention rows to named measurements.
- Calibration is tested from synthetic read-site cohorts to corrected token
  decisions, including the former any-head over-selection failure.
- Capture compatibility is tested against a minimal v3 directory.
- Tracing is tested against finite differences and independent autograd oracles.
- One small end-to-end fixture crosses the public pipeline interface.
