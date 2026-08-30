"""Render a Markdown document to a self-contained PDF.

Pipeline: Markdown -> styled, self-contained HTML -> PDF via a headless
Chromium browser (Chrome or Edge, whichever is installed). No LaTeX, no
system libraries beyond a browser that ships with Windows.

    # docs/results.md -> docs/results.pdf
    python scripts/build_report_pdf.py
    python scripts/build_report_pdf.py docs/architecture.md
    python scripts/build_report_pdf.py IN.md -o OUT.pdf
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

_BROWSER_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; max-width: 100%;
}
h1 { font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 4px; margin-top: 0; }
h2 { font-size: 15pt; border-bottom: 1px solid #ccc; padding-bottom: 3px;
     margin-top: 22px; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 16px; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin-top: 12px; page-break-after: avoid; }
p, li { orphans: 2; widows: 2; }
a { color: #0b5cad; text-decoration: none; }
code { font-family: "Cascadia Mono", Consolas, monospace; font-size: 9pt;
       background: #f2f2f2; padding: 1px 4px; border-radius: 3px; }
pre { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px;
      padding: 10px 12px; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.6pt; line-height: 1.45; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #cfcfcf; padding: 5px 8px; text-align: left;
         vertical-align: top; }
th { background: #f0f0f0; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
blockquote { border-left: 3px solid #b8b8b8; margin: 10px 0; padding: 2px 14px;
             color: #444; background: #fbfbfb; }
hr { border: none; border-top: 1px solid #ddd; margin: 20px 0; }
h2 { string-set: chapter content(text); }
""".strip()

HTML_SHELL = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style></head><body>{body}</body></html>
"""


def find_browser() -> str:
    for path in _BROWSER_CANDIDATES:
        if Path(path).exists():
            return path
    for name in ("chrome", "chromium", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("No Chrome/Edge/Chromium found to render the PDF.")


def md_to_html(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list", "md_in_html"],
    )
    return HTML_SHELL.format(title=md_path.stem, css=CSS, body=html_body)


def html_to_pdf(html: str, out_pdf: Path) -> None:
    browser = find_browser()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_pdf.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        html_file = Path(tmp) / "report.html"
        html_file.write_text(html, encoding="utf-8")
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={out_pdf}",
            html_file.as_uri(),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if not out_pdf.exists():
            sys.stderr.write(result.stdout + result.stderr)
            raise SystemExit(f"Browser did not produce {out_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="docs/results.md", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    md_path: Path = args.input
    if not md_path.exists():
        raise SystemExit(f"Not found: {md_path}")
    out_pdf: Path = args.output or md_path.with_suffix(".pdf")

    html_to_pdf(md_to_html(md_path), out_pdf)
    size_kb = out_pdf.stat().st_size / 1024
    print(f"Wrote {out_pdf} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
