"""Generate traces and inspect the attention used for decoding decisions."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect attention before a generated error")
    commands = parser.add_subparsers(dest="command", required=True)
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
    args = parser.parse_args(argv)
    if args.command == "sample":
        from decoding.sampling import SamplingConfig, SamplingExperiment

        SamplingExperiment(
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
        ).run()
    elif args.command == "inspect":
        from decoding.attention import AttentionAnalysis

        AttentionAnalysis(
            args.samples,
            args.output,
            args.trace,
            args.first_error,
            args.before,
            args.after,
            args.min_distance,
        ).run()
    else:
        from decoding.routes import RouteAnalysis

        RouteAnalysis(
            args.samples,
            args.cases,
            args.output,
            args.tokenizer,
            args.before,
            args.after,
            args.baseline,
        ).run()
    print(args.output)


if __name__ == "__main__":
    main()
