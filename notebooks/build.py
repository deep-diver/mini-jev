"""
Turn the `_source*.txt` files into Colab-ready notebooks.

The sources are plain text with `@@MD` / `@@CODE` markers rather than .ipynb so
that the prose stays reviewable in a diff. The Korean and English editions are
separate files because the code cells print in their own language; regenerate
after editing either.

    python3 notebooks/build.py           # both editions
    python3 notebooks/build.py --only en
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

EDITIONS = {
    "ko": ("_source.txt", "gemma3_270m_jev.ipynb"),
    "en": ("_source_en.txt", "gemma3_270m_jev_en.ipynb"),
}


def build(src, out):
    with open(src) as f:
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
    with open(out, "w") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    n_code = sum(c["cell_type"] == "code" for c in cells)
    print(f"{len(cells)} cells ({n_code} code) -> {os.path.basename(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(EDITIONS))
    args = ap.parse_args()

    for key, (src, out) in EDITIONS.items():
        if args.only and key != args.only:
            continue
        path = os.path.join(HERE, src)
        if not os.path.exists(path):
            print(f"skipping {key}: no {src}")
            continue
        build(path, os.path.join(HERE, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
