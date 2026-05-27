"""Convert a runnable .py script into a minimal .ipynb notebook (one cell)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def py_to_ipynb(py_path: Path, ipynb_path: Path | None = None):
    src = py_path.read_text()
    cells = [{
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ipynb_path or py_path.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        py_to_ipynb(Path(arg))
