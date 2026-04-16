# neurips2026_steering_ko

`paper/neurips2026_steering_v2`의 한국어 대응 원고다.

주의:

- 기존 `paper/neurips2026_ko` 폴더는 별도의 오래된 압축 논문 초안이며, 이 폴더가 그것을 대체하지 않는다.
- 이 폴더는 멀티-툴 선택과 ontology-guided query steering에 집중한 한국어 버전이다.

빌드:

1. `python scripts/build_figures.py`
2. `xelatex -interaction=nonstopmode main.tex`
3. `bibtex main`
4. `xelatex -interaction=nonstopmode main.tex`
5. `xelatex -interaction=nonstopmode main.tex`

