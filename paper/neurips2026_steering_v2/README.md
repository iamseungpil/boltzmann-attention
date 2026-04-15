# neurips2026_steering_v2

LaTeX project for the steering + KV-cache compression paper based on:

- `math/paper/benchmark_design/PAPER_DRAFT_v3.md`

The directory name is kept for continuity, but the generated PDF now tracks the
v3 draft (the Q-coverage pivot).

Build steps:

1. `python build_latex.py`
2. `xelatex -interaction=nonstopmode main.tex`
3. `xelatex -interaction=nonstopmode main.tex`

Artifacts:

- `content.tex`
- `sections/*.tex`
- `main.pdf`
