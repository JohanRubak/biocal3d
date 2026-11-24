"""Small I/O helpers shared by the study workers."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(name, **arrays)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
