"""
Turn notebooks/_source.txt into a Colab-ready .ipynb.

The source is kept as plain text with `@@MD` / `@@CODE` markers rather than a
.ipynb so the prose and the code stay reviewable in a diff; regenerate after
editing it.

    python3 notebooks/build.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "_source.txt")
OUT = os.path.join(HERE, "gemma3_270m_jev.ipynb")


def main():
    with open(SRC) as f:
        raw = f.read()

    cells = []
    for chunk in raw.split("@@")[1:]:
        kind, _, body = chunk.partition("\n")
        kind = kind.strip()
        body = body.strip("\n")
        if not body:
            continue
        cell = {
            "cell_type": "markdown" if kind == "MD" else "code",
            "metadata": {},
            "source": body.splitlines(keepends=True),
        }
        if kind == "CODE":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    with open(OUT, "w") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    n_code = sum(c["cell_type"] == "code" for c in cells)
    print(f"{len(cells)} cells ({n_code} code) -> {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
