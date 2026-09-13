"""Visible orchestration of the three reanchor stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reanchor.capture.protocol import AuditDataset
from reanchor.discovery.events import DiscoveryConfig, EventDiscovery
from reanchor.reporting.evaluation import ReportBuilder, ReportConfig
from reanchor.tracing.tracer import CausalTracer, TraceConfig


@dataclass(frozen=True)
class PipelineConfig:
    capture: Path
    output: Path
    command: str = "run"
    discovery: DiscoveryConfig = DiscoveryConfig()
    tracing: TraceConfig = TraceConfig()
    reporting: ReportConfig = ReportConfig()

    def __post_init__(self):
        if self.command not in {"discover", "trace", "report", "run"}:
            raise ValueError("command must be discover, trace, report, or run")


class ReanchorPipeline:
    """Execute requested stages without hiding their inputs or outputs."""

    def __init__(self, config: PipelineConfig, *, progress=None):
        self.config = config
        self.progress = progress

    def run(self) -> dict:
        dataset = AuditDataset(self.config.capture)
        result = {}
        if self.config.command in {"discover", "run"}:
            result["discovery"] = EventDiscovery(self.config.discovery).run(
                dataset, self.config.output
            )
        if self.config.command in {"trace", "run"}:
            result["tracing"] = CausalTracer(self.config.tracing, progress=self.progress).run(
                dataset, self.config.output
            )
        if self.config.command in {"report", "run"}:
            result["report"] = ReportBuilder(self.config.reporting).run(dataset, self.config.output)
        return result
