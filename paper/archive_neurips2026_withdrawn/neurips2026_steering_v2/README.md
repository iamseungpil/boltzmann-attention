# neurips2026_steering_v2

Curated LaTeX project for the steering paper centered on multi-tool selection.
Despite the legacy directory name, this project is no longer a KV-cache
compression paper. The current manuscript makes a narrower claim:

- multi-tool prompts require coverage, not repeated emphasis of one facet
- ontology-guided query-side suppression is mildly positive on that regime
- stationary key-side amplification is negative on the same regime
- ontology specificity is more decisive than raw gain magnitude

Source layout:

- `main.tex`: paper preamble and title page
- `content.tex`: section include order
- `sections/`: main manuscript text
- `figures/`: generated plots used in the paper
- `refs.bib`: bibliography
- `FACT_BASE.md`: verified result anchors used in the manuscript

Build:

1. `python scripts/build_figures.py`
2. `xelatex -interaction=nonstopmode main.tex`
3. `bibtex main`
4. `xelatex -interaction=nonstopmode main.tex`
5. `xelatex -interaction=nonstopmode main.tex`

The repository also contains other paper directories. This folder is the
English steering manuscript; do not confuse it with the older Korean
compression draft under `paper/neurips2026_ko`.
