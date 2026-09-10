# Reanchor: factorial constraint control

This repository tests one mechanism: after a model emits a factual commitment,
does the generated prefix suppress the source condition's causal control over a
related continuation?

It is an independent experiment. The retired attention-event discovery,
message-DAG, JVP tracing, feature registry, and duplicate evaluation paths have
been removed.

## Method

Each event defines two minimally different source worlds. World A supports
option A and world B supports option B. The model scores the same A-minus-B
answer margin before a commitment and after forcing commitment A or B:

| source world | forced A | forced B |
| --- | ---: | ---: |
| A | `M_AA` | `M_AB` |
| B | `M_BA` | `M_BB` |

Only four factorial coordinates are reported:

```text
source_onset = (onset_A - onset_B) / 2
source_followup = (M_AA + M_AB - M_BA - M_BB) / 4
prefix_followup = (M_AA - M_AB + M_BA - M_BB) / 4
source_prefix_coupling = |M_AA - M_AB - M_BA + M_BB| / 4
```

This separates source influence, visible-prefix influence, and their
interaction. The retired `control_erosion` statistic mixed these terms: its
formula was twice the prefix main effect, not isolated source erosion.

The candidate failure is **prefix-induced constraint shielding**:
`source_onset` is strong, but `source_followup` becomes weak while
`prefix_followup` dominates. This is an output-level intervention result; it is
not inferred from attention distance.

## Execution path

```text
main.py
  -> ConstraintControlExperiment.run()
       -> load_events()          src/reanchor/data.py
       -> CausalLMScorer.score() src/reanchor/model.py
       -> compute_effects()      src/reanchor/effects.py
       -> events.jsonl + summary.json
```

The experiment uses one batched forward pass per event to score six contexts
and twelve candidate continuations. It saves the exact input snapshot,
model/config provenance, raw candidate log probabilities, six margins, four
effects, and aggregate means.

## Input

Every JSONL line must contain exactly:

```json
{
  "schema": "reanchor/counterfactual-event@1",
  "event_id": "question-17/fact-0",
  "source_id": "question-17",
  "split": "train",
  "relation": "temporal",
  "prompt_a": "The source says Tuesday ...",
  "prompt_b": "The source says Thursday ...",
  "answer_prefix": "The train departs on ",
  "option_a": "Tuesday",
  "option_b": "Thursday",
  "followup": ". Re-checking the source, it departs on "
}
```

Labels are forbidden from measurement input. `source_id` is the base-item
group used to prevent split leakage. `data/pilot_events.jsonl` contains five
small mechanism smoke tests, not a confirmatory benchmark.

RAGTruth labels do not by themselves define minimal A/B source worlds. A later
RAGTruth experiment must freeze factual events, candidate answers, world edits,
and source-disjoint splits before inspecting hallucination labels.

## Run

From the repository root on the GPU server:

```bash
bash scripts/run_pilot.sh
```

Optional positional arguments are input, output, local model, and device. The
defaults are:

```text
input   data/pilot_events.jsonl
output  outputs/constraint_control_pilot
model   /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct
device  cuda:0
```

The model is loaded with `local_files_only=True`, so the run never attempts to
access the gated Hugging Face repository. The output directory must be empty.

## Claim boundary

The 2x2 intervention is an identification instrument, not by itself a new model
architecture. A mechanism claim requires consistent directional effects across
relation types, source-disjoint data, A/B swaps, lexical placebos, and multiple
model families. A training method should be added only after this pilot finds a
replicable source-recovery deficit.
