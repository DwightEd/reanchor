"""Command-line boundary for the reanchor pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reanchor.discovery.events import DiscoveryConfig
from reanchor.pipeline import PipelineConfig, ReanchorPipeline
from reanchor.reporting.evaluation import ReportConfig
from reanchor.tracing.tracer import TraceConfig


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="reanchor",
        description="Discover calibrated reanchor episodes and trace their causal messages.",
    )
    command.add_argument("command", choices=("discover", "trace", "report", "run"))
    command.add_argument("--capture", type=Path, required=True, help="attention_audit_v3 root")
    command.add_argument("--output", type=Path, required=True, help="separate run directory")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--window", type=int, default=10)
    command.add_argument("--local-floor", type=float, default=0.50)
    command.add_argument("--site-gain-floor", type=float, default=0.10)
    command.add_argument("--broad-gain-floor", type=float, default=0.05)
    command.add_argument("--broad-head-fraction", type=float, default=0.25)
    command.add_argument("--position-bins", type=int, default=4)
    command.add_argument("--alpha", type=float, default=0.05)
    command.add_argument("--min-calibration-sources", type=int, default=32)
    command.add_argument("--episode-gap", type=int, default=1)
    command.add_argument("--calibration-split", default="train")
    command.add_argument("--query-chunk", type=int, default=8)
    command.add_argument("--event-batch", type=int, default=2)
    command.add_argument("--no-edges", action="store_true")
    command.add_argument(
        "--contrasts",
        type=Path,
        help="optional JSON mapping sample keys to explicit candidate-token contrasts",
    )
    command.add_argument("--bootstrap", type=int, default=1000)
    return command


def main(argv: list[str] | None = None) -> None:
    arguments = parser().parse_args(argv)
    discovery = DiscoveryConfig(
        window=arguments.window,
        local_floor=arguments.local_floor,
        site_gain_floor=arguments.site_gain_floor,
        broad_gain_floor=arguments.broad_gain_floor,
        broad_head_fraction=arguments.broad_head_fraction,
        position_bins=arguments.position_bins,
        family_alpha=arguments.alpha,
        min_calibration_sources=arguments.min_calibration_sources,
        episode_gap=arguments.episode_gap,
        calibration_split=arguments.calibration_split,
        device=arguments.device,
        query_chunk=arguments.query_chunk,
    )
    tracing = TraceConfig(
        device=arguments.device,
        query_chunk=arguments.query_chunk,
        event_batch=arguments.event_batch,
        save_edges=not arguments.no_edges,
        contrast_file=str(arguments.contrasts) if arguments.contrasts else None,
    )
    config = PipelineConfig(
        capture=arguments.capture,
        output=arguments.output,
        command=arguments.command,
        discovery=discovery,
        tracing=tracing,
        reporting=ReportConfig(bootstrap=arguments.bootstrap),
    )

    def progress(message):
        print(message, file=sys.stderr, flush=True)

    result = ReanchorPipeline(config, progress=progress).run()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
