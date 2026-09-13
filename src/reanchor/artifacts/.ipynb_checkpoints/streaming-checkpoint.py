"""Atomic streaming support for large NPZ edge artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


class AtomicArtifact:
    def __init__(self, destination: str | Path):
        self.destination = Path(destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            dir=self.destination.parent,
            prefix=f".{self.destination.stem}.",
            suffix=self.destination.suffix,
            delete=False,
        ) as handle:
            self.temporary = Path(handle.name)

    def commit(self) -> None:
        os.replace(self.temporary, self.destination)

    def discard(self) -> None:
        self.temporary.unlink(missing_ok=True)


def write_array(archive, name: str, value) -> None:
    """Write one unpickled NPY member without buffering the whole NPZ."""

    with archive.open(name + ".npy", "w", force_zip64=True) as stream:
        np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)
