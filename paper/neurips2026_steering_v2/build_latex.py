#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent.parent / "math" / "paper" / "benchmark_design" / "PAPER_DRAFT_v2.md"
SECTIONS_DIR = ROOT / "sections"
CONTENT_TEX = ROOT / "content.tex"


def split_math(text: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    pattern = re.compile(r"(\$\$.*?\$\$|\$.*?\$)", re.DOTALL)
    last = 0
    for match in pattern.finditer(text):
        if match.start() > last:
            parts.append(("text", text[last:match.start()]))
        parts.append(("math", match.group(0)))
        last = match.end()
    if last < len(text):
        parts.append(("text", text[last:]))
    return parts


def escape_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "^": r"\^{}",
        "{": r"\{",
        "}": r"\}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def convert_inline(text: str) -> str:
    chunks = []
    for kind, part in split_math(text):
        if kind == "math":
            chunks.append(part)
            continue
        code_map: dict[str, str] = {}

        def escape_code(code: str) -> str:
            semantic_replacements = {
                "‖": "||",
                "⊥": "perp",
                "⇒": "=>",
                "≤": "<=",
                "≥": ">=",
                "∈": "in",
                "ρ": "rho",
                "α": "alpha",
                "β": "beta",
                "γ": "gamma",
                "ε": "epsilon",
                "τ": "tau",
                "₁": "_1",
                "₂": "_2",
                "²": "^2",
                "⁴": "^4",
                "⁷": "^7",
                "⁸": "^8",
                "⁻": "^-",
                "ᵀ": "^T",
            }
            latex_replacements = {
                "\\": r"\textbackslash{}",
                "{": r"\{",
                "}": r"\}",
                "_": r"\_",
                "^": r"\^{}",
                "&": r"\&",
                "%": r"\%",
                "#": r"\#",
            }
            for src, dst in semantic_replacements.items():
                code = code.replace(src, dst)
            for src, dst in latex_replacements.items():
                code = code.replace(src, dst)
            return code

        def stash_code(match: re.Match[str]) -> str:
            key = f"@@CODE{len(code_map)}@@"
            code_map[key] = r"\texttt{" + escape_code(match.group(1)) + "}"
            return key

        part = re.sub(r"`([^`]+)`", stash_code, part)
        part = escape_text(part)
        part = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", part)
        part = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\emph{\1}", part)
        for key, value in code_map.items():
            part = part.replace(key, value)
        chunks.append(part)
    return "".join(chunks)


def heading_title(raw: str) -> str:
    return convert_inline(raw.strip())


def slugify(index: int, title: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return f"{index:02d}_{base or 'section'}.tex"


def is_table_line(line: str) -> bool:
    line = line.strip()
    return line.startswith("|") and line.endswith("|")


def split_table_row(line: str) -> list[str]:
    body = line.strip().strip("|")
    cells: list[str] = []
    current: list[str] = []
    in_math = False
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "$":
            in_math = not in_math
            current.append(ch)
        elif ch == "|" and not in_math:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
        i += 1
    cells.append("".join(current).strip())
    return cells


def parse_table(block: list[str]) -> str:
    rows = []
    for line in block:
        line = line.strip()
        if re.fullmatch(r"\|?[\s:-|]+\|?", line):
            continue
        raw_cells = split_table_row(line)
        if raw_cells and all(re.fullmatch(r":?-+:?", cell.replace(" ", "")) for cell in raw_cells):
            continue
        cells = [convert_inline(cell.strip()) for cell in raw_cells]
        rows.append(cells)
    if not rows:
        return ""
    ncols = max(len(row) for row in rows)
    colspec = " | ".join(["X"] * ncols)
    out = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\begin{{tabularx}}{{\linewidth}}{{{colspec}}}",
        r"\toprule",
    ]
    header = rows[0] + [""] * (ncols - len(rows[0]))
    out.append(" & ".join(header) + r" \\")
    out.append(r"\midrule")
    for row in rows[1:]:
        padded = row + [""] * (ncols - len(row))
        out.append(" & ".join(padded) + r" \\")
    out.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    return "\n".join(out)


def parse_blockquote(block: list[str]) -> str:
    body = "\n".join(convert_inline(line[2:] if line.startswith("> ") else line[1:]) for line in block)
    return "\n".join([r"\begin{quote}", body, r"\end{quote}"])


def parse_list(block: list[str]) -> str:
    numbered = bool(re.match(r"\d+\.\s", block[0].strip()))
    env = "enumerate" if numbered else "itemize"
    out = [rf"\begin{{{env}}}"]
    for line in block:
        stripped = line.strip()
        stripped = re.sub(r"^[-*]\s+", "", stripped)
        stripped = re.sub(r"^\d+\.\s+", "", stripped)
        out.append(r"\item " + convert_inline(stripped))
    out.append(rf"\end{{{env}}}")
    return "\n".join(out)


def parse_code(block: list[str]) -> str:
    body = "\n".join(block[1:-1])
    return "\n".join([r"\begin{verbatim}", body, r"\end{verbatim}"])


def parse_paragraph(block: list[str]) -> str:
    text = " ".join(line.strip() for line in block)
    return convert_inline(text) + "\n"


def render_body(lines: list[str], allow_section_heading: bool) -> str:
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped == "---":
            i += 1
            continue
        if stripped.startswith("### "):
            out.append(r"\subsection{" + heading_title(stripped[4:]) + "}")
            i += 1
            continue
        if stripped.startswith("#### "):
            out.append(r"\subsubsection{" + heading_title(stripped[5:]) + "}")
            i += 1
            continue
        if allow_section_heading and stripped.startswith("## "):
            out.append(r"\section{" + heading_title(stripped[3:]) + "}")
            i += 1
            continue
        if stripped.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            if j < len(lines):
                j += 1
            out.append(parse_code(lines[i:j]))
            i = j
            continue
        if is_table_line(stripped):
            j = i
            while j < len(lines) and is_table_line(lines[j]):
                j += 1
            out.append(parse_table(lines[i:j]))
            i = j
            continue
        if stripped.startswith(">"):
            j = i
            while j < len(lines) and lines[j].strip().startswith(">"):
                j += 1
            out.append(parse_blockquote(lines[i:j]))
            i = j
            continue
        if re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            j = i
            while j < len(lines):
                cur = lines[j].strip()
                if re.match(r"^[-*]\s+", cur) or re.match(r"^\d+\.\s+", cur):
                    j += 1
                else:
                    break
            out.append(parse_list(lines[i:j]))
            i = j
            continue
        j = i
        para: list[str] = []
        while j < len(lines):
            cur = lines[j].strip()
            if (
                not cur
                or cur == "---"
                or cur.startswith("## ")
                or cur.startswith("### ")
                or cur.startswith("#### ")
                or cur.startswith("```")
                or cur.startswith(">")
                or is_table_line(cur)
                or re.match(r"^[-*]\s+", cur)
                or re.match(r"^\d+\.\s+", cur)
            ):
                break
            para.append(lines[j])
            j += 1
        out.append(parse_paragraph(para))
        i = j
    return "\n\n".join(part for part in out if part.strip()) + "\n"


def load_sections() -> list[tuple[str, list[str]]]:
    text = SOURCE.read_text(encoding="utf-8")
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = "_frontmatter"
    current_lines: list[str] = []
    for line in lines:
        if line.startswith("## "):
            sections.append((current_title, current_lines))
            current_title = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_title, current_lines))
    return sections


def main() -> None:
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    sections = load_sections()
    content_inputs: list[str] = []
    for idx, (title, body_lines) in enumerate(sections):
        if title == "_frontmatter":
            continue
        filename = slugify(idx, title)
        out_path = SECTIONS_DIR / filename
        if title == "Abstract":
            body = render_body(body_lines, allow_section_heading=False).strip()
            tex = "\n".join([r"\begin{abstract}", body, r"\end{abstract}", ""])
        elif title == "Appendices":
            body = render_body(body_lines, allow_section_heading=False)
            tex = "\n".join([r"\appendix", r"\section{Appendices}", body])
        else:
            body = render_body(body_lines, allow_section_heading=False)
            tex = "\n".join([r"\section{" + heading_title(title) + "}", body])
        out_path.write_text(tex, encoding="utf-8")
        content_inputs.append(rf"\input{{sections/{filename}}}")
    CONTENT_TEX.write_text("\n".join(content_inputs) + "\n", encoding="utf-8")
    print(f"Wrote {len(content_inputs)} sections from {SOURCE}")


if __name__ == "__main__":
    main()
