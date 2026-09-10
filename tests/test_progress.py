from io import StringIO


def test_tqdm_progress_tracks_items_and_updates_the_active_bar(monkeypatch):
    bars = []

    class FakeBar:
        def __init__(self, items, **options):
            self.items = items
            self.options = options
            self.details = []
            self.closed = False
            bars.append(self)

        def __iter__(self):
            yield from self.items

        def set_postfix_str(self, message, *, refresh):
            self.details.append((message, refresh))

        def close(self):
            self.closed = True

    import reanchor.progress as module

    monkeypatch.setattr(module, "tqdm", FakeBar)
    progress = module.TqdmProgress(file=StringIO())

    visited = []
    for item in progress.track([1, 2], description="trace samples"):
        visited.append(item)
        progress.detail(f"sample={item}")

    assert visited == [1, 2]
    assert bars[0].options["desc"] == "trace samples"
    assert bars[0].options["unit"] == "sample"
    assert bars[0].details == [("sample=1", True), ("sample=2", True)]
    assert bars[0].closed
