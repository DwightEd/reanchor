from pathlib import Path

from reanchor.discovery.events import DiscoveryConfig
from reanchor.pipeline import PipelineConfig, ReanchorPipeline
from reanchor.reporting.evaluation import ReportConfig
from reanchor.tracing.tracer import TraceConfig


def test_pipeline_keeps_the_default_execution_path_linear(monkeypatch, tmp_path):
    calls = []

    class Dataset:
        def __init__(self, path):
            calls.append(("dataset", Path(path)))

    class Workflow:
        def __init__(self, name):
            self.name = name

        def run(self, dataset, output):
            calls.append((self.name, Path(output)))
            return {"stage": self.name}

    import reanchor.pipeline as module

    monkeypatch.setattr(module, "AuditDataset", Dataset)
    monkeypatch.setattr(module, "EventDiscovery", lambda *args, **kwargs: Workflow("discover"))
    monkeypatch.setattr(module, "CausalTracer", lambda *args, **kwargs: Workflow("trace"))
    monkeypatch.setattr(module, "ReportBuilder", lambda *args, **kwargs: Workflow("report"))
    config = PipelineConfig(
        capture=tmp_path / "capture",
        output=tmp_path / "run",
        command="run",
        discovery=DiscoveryConfig(device="cpu"),
        tracing=TraceConfig(device="cpu"),
        reporting=ReportConfig(bootstrap=0),
    )

    result = ReanchorPipeline(config).run()

    assert [name for name, _ in calls] == ["dataset", "discover", "trace", "report"]
    assert set(result) == {"discovery", "tracing", "report"}


def test_pipeline_runs_mechanism_audit_as_an_explicit_post_trace_stage(monkeypatch, tmp_path):
    calls = []

    class Dataset:
        def __init__(self, path):
            calls.append(("dataset", Path(path)))

    class Auditor:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, dataset, output):
            calls.append(("audit", Path(output)))
            return {"stage": "audit"}

    import reanchor.pipeline as module

    monkeypatch.setattr(module, "AuditDataset", Dataset)
    monkeypatch.setattr(module, "MechanismAuditor", Auditor)
    config = PipelineConfig(
        capture=tmp_path / "capture",
        output=tmp_path / "run",
        command="audit",
    )

    result = ReanchorPipeline(config).run()

    assert [name for name, _ in calls] == ["dataset", "audit"]
    assert result == {"mechanism": {"stage": "audit"}}
