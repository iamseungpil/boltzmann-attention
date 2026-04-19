# FOKVQ NeurIPS 2026 Korean Draft

이 디렉터리는 FOKVQ 논문의 NeurIPS 2026 제출 준비를 위한 한국어 내부 LaTeX 초안이다.

## 구성

- `main.tex`: 메인 엔트리 포인트
- `sections/`: 섹션별 원고
- `figures/`: 논문 그림
- `refs.bib`: 참고문헌 BibTeX 데이터베이스
- `FACT_BASE.md`: 초안에 반영한 검증 사실 요약

## 빌드

```bash
xelatex -interaction=nonstopmode main.tex
bibtex main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

## 현재 상태

- 한국어 내부 초안 PDF 빌드 완료
- 정리, 보조정리, 증명 구조 반영 완료
- 표준 WikiText-2 PPL 및 외부 baseline 표는 추후 교체 항목으로 유지
