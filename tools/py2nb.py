"""Converte um script em formato percent (# %% / # %% [markdown]) em .ipynb e, opcionalmente, executa.

Escrever notebooks como .py é mais rápido, gera diffs legíveis e evita editar JSON à mão.

Uso:
    python tools/py2nb.py notebooks/src/01_eda.py notebooks/01_eda.ipynb [--execute]
"""
import argparse
import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

CELL = re.compile(r"^# %%(.*)$", re.M)


def convert(src: Path) -> nbformat.NotebookNode:
    text = src.read_text(encoding="utf-8")
    parts = CELL.split(text)
    cells = []
    for header, body in zip(parts[1::2], parts[2::2]):
        body = body.strip("\n")
        if "[markdown]" in header:
            md = "\n".join(re.sub(r"^# ?", "", ln) for ln in body.splitlines())
            cells.append(new_markdown_cell(md))
        elif body.strip():
            cells.append(new_code_cell(body))
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "nslkdd", "display_name": "Python (nslkdd)",
                                 "language": "python"}
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    nb = convert(args.src)
    if args.execute:
        from nbconvert.preprocessors import ExecutePreprocessor

        ExecutePreprocessor(timeout=args.timeout, kernel_name="nslkdd").preprocess(
            nb, {"metadata": {"path": str(args.dst.parent.resolve())}})
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, args.dst)
    print(f"{args.dst} ({len(nb.cells)} células){' executado' if args.execute else ''}")


if __name__ == "__main__":
    main()
