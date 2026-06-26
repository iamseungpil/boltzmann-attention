#!/usr/bin/env bash
# 3-pass LaTeX build for the section-based project (academic-latex-pipeline Phase 2.5).
# Usage: ./build_and_compile.sh   (run from this folder)
set -e
JOB=main
pdflatex -interaction=nonstopmode "$JOB.tex"
bibtex "$JOB" || true          # bibtex is non-fatal on first run if no \cite yet
pdflatex -interaction=nonstopmode "$JOB.tex"
pdflatex -interaction=nonstopmode "$JOB.tex"
echo "Built $JOB.pdf"
