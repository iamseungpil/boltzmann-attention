#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_pdf.py — concat markdown file(s) -> HTML -> PDF (xhtml2pdf, pure-python).
usage: py -3 build_pdf.py out.pdf in1.md [in2.md ...]
"""
import sys, markdown
from xhtml2pdf import pisa

CSS = """
@page { size: letter portrait; margin: 1.8cm 1.9cm 2.0cm 1.9cm; }
body { font-family: "Times New Roman", serif; font-size: 9.6pt; line-height: 1.34; color: #111; }
h1 { font-size: 16pt; text-align: center; margin: 0 0 4pt 0; line-height: 1.18; }
h2 { font-size: 11.5pt; margin: 12pt 0 3pt 0; border-bottom: 0.6pt solid #999; padding-bottom: 1pt; }
h3 { font-size: 10.2pt; margin: 9pt 0 2pt 0; font-style: italic; }
h4 { font-size: 9.8pt; margin: 7pt 0 2pt 0; }
p  { margin: 0 0 5pt 0; text-align: justify; }
em { font-style: italic; } strong { font-weight: bold; }
ul, ol { margin: 0 0 5pt 14pt; padding: 0; }
li { margin: 0 0 1.5pt 0; }
blockquote { margin: 4pt 0 6pt 0; padding: 3pt 7pt; background: #f4f4f4;
             border-left: 2.4pt solid #bbb; font-size: 9.0pt; color: #333; }
code { font-family: "Courier New", monospace; font-size: 8.6pt; background: #f0f0f0; }
pre  { font-family: "Courier New", monospace; font-size: 8.4pt; background: #f4f4f4;
       padding: 4pt 6pt; margin: 4pt 0; }
hr { border: none; border-top: 0.5pt solid #ccc; margin: 7pt 0; }
table { border-collapse: collapse; margin: 5pt 0 7pt 0; width: 100%; }
th, td { border: 0.5pt solid #999; padding: 2.2pt 4pt; font-size: 8.7pt; text-align: left;
         vertical-align: top; }
th { background: #e9e9e9; font-weight: bold; }
a { color: #1a3e7a; text-decoration: none; }
"""


def main():
    out = sys.argv[1]
    md_text = "\n\n".join(open(f, encoding="utf-8").read() for f in sys.argv[2:])
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    with open(out, "wb") as fh:
        res = pisa.CreatePDF(html, dest=fh, encoding="utf-8")
    print("PDF errors:", res.err, "->", out)
    sys.exit(1 if res.err else 0)


if __name__ == "__main__":
    main()
