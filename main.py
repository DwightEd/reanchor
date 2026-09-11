"""Generate traces and inspect the attention used for decoding decisions."""

import argparse
from pathlib import Path


def capture_arguments(commands):
    sample = commands.add_parser("sample", help="sample answers and capture attention/logits")
    sample.add_argument("--dataset", type=Path, required=True)
    sample.add_argument("--source-ids", nargs="+", required=True)
    sample.add_argument("--model", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    sample.add_argument("--max-new-tokens", type=int, default=512)
    sample.add_argument("--temperature", type=float, default=0.7)
    sample.add_argument("--top-p", type=float, default=0.9)
    sample.add_argument("--device", default="cuda:0")
    sample.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    states = commands.add_parser("states", help="capture states on the saved token sequence")
    states.add_argument("--samples", type=Path, required=True)
    states.add_argument("--output", type=Path, required=True)
    states.add_argument("--device", help="defaults to the original sampling device")
    states.add_argument("--atol", type=float, default=1e-4, help="maximum native logit difference")


def analysis_arguments(commands):
    inspect = commands.add_parser(
        "inspect", help="write token choices and concrete attention shifts"
    )
    inspect.add_argument("--samples", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    inspect.add_argument("--trace", help="one NPZ filename from samples.jsonl")
    inspect.add_argument("--first-error", type=int, help="manually identified response token index")
    inspect.add_argument("--before", type=int, default=16)
    inspect.add_argument("--after", type=int, default=0)
    inspect.add_argument(
        "--min-distance", type=int, default=16, help="minimum query-to-key token lag"
    )
    routes = commands.add_parser("routes", help="analyze source attention in concrete examples")
    routes.add_argument("--samples", type=Path, required=True)
    routes.add_argument("--cases", type=Path, required=True, help="CSV: source_id,seed,focus")
    routes.add_argument("--output", type=Path, required=True)
    routes.add_argument("--tokenizer", type=Path, help="defaults to the captured model path")
    routes.add_argument("--before", type=int, default=16)
    routes.add_argument("--after", type=int, default=8)
    routes.add_argument("--baseline", type=int, default=16, help="previous steps for shift median")
    decisions = commands.add_parser("decisions", help="inspect fixed decision windows from NPZ")
    decisions.add_argument("--samples", type=Path, required=True)
    decisions.add_argument("--cases", type=Path, required=True)
    decisions.add_argument("--output", type=Path, required=True)
    revisits = commands.add_parser("revisits", help="scan all saved decisions without case labels")
    revisits.add_argument("--samples", type=Path, required=True)
    revisits.add_argument("--output", type=Path, required=True)
    revisits.add_argument("--window", type=int, default=16, help="past steps for event baseline")
    revisits.add_argument("--quantile", type=float, default=0.95, help="past-score event quantile")
    revisits.add_argument("--context", type=int, default=4, help="inspect steps around each event")
    revisits.add_argument("--hops", type=int, default=3, help="maximum attention path length")
    revisits.add_argument("--states", type=Path, help="fixed-prefix states for relation readout")
    compare = commands.add_parser("compare", help="join case windows to full-stream readouts")
    compare.add_argument("--analysis", type=Path, required=True)
    compare.add_argument("--cases", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)


def experiment(args):
    if args.command == "sample":
        from decoding.sampling import SamplingConfig, SamplingExperiment

        return SamplingExperiment(
            SamplingConfig(
                dataset=args.dataset,
                source_ids=tuple(args.source_ids),
                model=args.model,
                output=args.output,
                seeds=tuple(args.seeds),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
                dtype=args.dtype,
            )
        )
    elif args.command == "inspect":
        from decoding.attention import AttentionAnalysis

        return AttentionAnalysis(
            args.samples,
            args.output,
            args.trace,
            args.first_error,
            args.before,
            args.after,
            args.min_distance,
        )
    elif args.command == "routes":
        from decoding.routes import RouteAnalysis

        return RouteAnalysis(
            args.samples,
            args.cases,
            args.output,
            args.tokenizer,
            args.before,
            args.after,
            args.baseline,
        )
    elif args.command == "decisions":
        from decoding.decisions import DecisionInspection

        return DecisionInspection(args.samples, args.cases, args.output)
    elif args.command == "states":
        from decoding.fixed_prefix import FixedPrefixStates

        return FixedPrefixStates(args.samples, args.output, args.device, args.atol)
    elif args.command == "compare":
        from decoding.window_comparison import WindowComparison

        return WindowComparison(args.analysis, args.cases, args.output)
    else:
        from decoding.revisits import RevisitAnalysis

        return RevisitAnalysis(
            args.samples,
            args.output,
            args.window,
            args.quantile,
            args.context,
            args.hops,
            args.states,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect reading decisions and source relations")
    commands = parser.add_subparsers(dest="command", required=True)
    capture_arguments(commands)
    analysis_arguments(commands)
    args = parser.parse_args(argv)
    experiment(args).run()
    print(args.output)


if __name__ == "__main__":
    main()
