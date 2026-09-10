"""Terminal progress display for long-running pipeline stages."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm.auto import tqdm

Item = TypeVar("Item")


class TqdmProgress:
    """Render sample progress while keeping inner-loop detail on one line."""

    def __init__(self, *, file=None):
        self.file = sys.stderr if file is None else file
        self._active = None

    def track(
        self,
        items: Iterable[Item],
        *,
        description: str,
        total: int | None = None,
    ) -> Iterator[Item]:
        bar = tqdm(
            items,
            desc=description,
            total=total,
            unit="sample",
            dynamic_ncols=True,
            file=self.file,
        )
        previous = self._active
        self._active = bar
        try:
            yield from bar
        finally:
            self._active = previous
            bar.close()

    def detail(self, message: str) -> None:
        if self._active is None:
            tqdm.write(message, file=self.file)
            return
        self._active.set_postfix_str(message, refresh=True)
