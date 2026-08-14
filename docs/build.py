"""Render the repository's Markdown documentation into a static site.

The sources stay where they are -- this script reads them from the repository
root so the published site can never drift from the files under review. Run it
from the repository root:

    python docs/build.py --out _site
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Page:
    source: str  # path relative to the repository root
    output: str  # file name written into the output directory
    title: str  # label used in the sidebar
    blurb: str  # one line, used on the landing card and in <meta>


PAGES: tuple[Page, ...] = (
    Page(
        "README.md",
        "index.html",
        "Overview",
        "Deterministic, saturation-aware snowballing for systematic literature reviews.",
    ),
    Page(
        "COLAB.md",
        "colab.html",
        "Run it in Colab",
        "Install and run a review in a Google Colab notebook, with nothing to set up locally.",
    ),
    Page(
        "SPEC.md",
        "spec.html",
        "Specification",
        "The implementation specification: data model, providers, stopping rules, determinism layer.",
    ),
    Page(
        "VALIDATION_REPORT.md",
        "validation.html",
        "Validation report",
        "What was measured against known ground truth, including the results that were unfavourable.",
    ),
    Page(
        "validation/README.md",
        "validation-code.html",
        "Validation code",
        "The offline, deterministic harness that reproduces every number in the validation report.",
    ),
)

CSS = """
:root {
  --bg: #fdfcfa;
  --surface: #ffffff;
  --surface-alt: #f5f2ec;
  --text: #1c1a17;
  --text-muted: #5d574e;
  --border: #e2ddd3;
  --accent: #8a5a2b;
  --accent-soft: #f0e7db;
  --code-bg: #f5f2ec;
  --sidebar-w: 17rem;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #161513;
    --surface: #1d1c19;
    --surface-alt: #232120;
    --text: #ece8e1;
    --text-muted: #a8a196;
    --border: #322f2b;
    --accent: #d9a066;
    --accent-soft: #2a2521;
    --code-bg: #232120;
  }
}

* { box-sizing: border-box; }

html { scroll-behavior: smooth; scroll-padding-top: 1.5rem; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 16px;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}

a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

.layout { display: flex; align-items: flex-start; max-width: 84rem; margin: 0 auto; }

/* ---- sidebar ---- */

.sidebar {
  width: var(--sidebar-w);
  flex: 0 0 var(--sidebar-w);
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
  padding: 2rem 1.25rem;
  border-right: 1px solid var(--border);
}

.brand { display: block; margin-bottom: 0.25rem; font-size: 1.15rem; font-weight: 650; color: var(--text); letter-spacing: -0.01em; }
.brand:hover { text-decoration: none; color: var(--accent); }
.brand-sub { margin: 0 0 1.75rem; font-size: 0.8rem; color: var(--text-muted); line-height: 1.45; }

.nav-label {
  margin: 1.5rem 0 0.5rem;
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--text-muted);
}
.nav-label:first-of-type { margin-top: 0; }

.nav { list-style: none; margin: 0; padding: 0; }
.nav li { margin: 0; }
.nav a {
  display: block;
  padding: 0.32rem 0.6rem;
  margin-left: -0.6rem;
  border-radius: 5px;
  color: var(--text-muted);
  font-size: 0.9rem;
}
.nav a:hover { background: var(--surface-alt); color: var(--text); text-decoration: none; }
.nav a.current { background: var(--accent-soft); color: var(--accent); font-weight: 550; }

/* per-page table of contents */
.toc { list-style: none; margin: 0; padding: 0; border-left: 1px solid var(--border); }
.toc li { margin: 0; }
.toc a {
  display: block;
  padding: 0.22rem 0 0.22rem 0.75rem;
  margin-left: -1px;
  border-left: 2px solid transparent;
  color: var(--text-muted);
  font-size: 0.84rem;
  line-height: 1.4;
}
.toc a:hover { color: var(--text); border-left-color: var(--border); text-decoration: none; }

/* ---- content ---- */

.content { flex: 1 1 auto; min-width: 0; padding: 2.5rem 3rem 6rem; }

.content h1, .content h2, .content h3, .content h4 {
  line-height: 1.25;
  letter-spacing: -0.015em;
  scroll-margin-top: 1.5rem;
}
.content h1 { margin: 0 0 1.5rem; font-size: 2.1rem; font-weight: 680; }
.content h2 {
  margin: 3rem 0 1rem;
  padding-top: 1.25rem;
  border-top: 1px solid var(--border);
  font-size: 1.4rem;
  font-weight: 640;
}
.content h3 { margin: 2rem 0 0.75rem; font-size: 1.1rem; font-weight: 620; }
.content h4 { margin: 1.5rem 0 0.5rem; font-size: 0.97rem; font-weight: 620; color: var(--text-muted); }

.content p { margin: 0 0 1.1rem; }
.content ul, .content ol { margin: 0 0 1.1rem; padding-left: 1.4rem; }
.content li { margin: 0.3rem 0; }
.content li > p { margin-bottom: 0.5rem; }

.content strong { font-weight: 640; }

.content hr { margin: 2.5rem 0; border: 0; border-top: 1px solid var(--border); }

.content blockquote {
  margin: 1.5rem 0;
  padding: 0.85rem 1.25rem;
  border-left: 3px solid var(--accent);
  background: var(--surface-alt);
  border-radius: 0 6px 6px 0;
  color: var(--text-muted);
}
.content blockquote p:last-child { margin-bottom: 0; }

code, pre, kbd, samp {
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}

.content :not(pre) > code {
  padding: 0.13em 0.38em;
  border-radius: 4px;
  background: var(--code-bg);
  border: 1px solid var(--border);
  font-size: 0.86em;
  word-break: break-word;
}

.content pre {
  margin: 0 0 1.3rem;
  padding: 1rem 1.15rem;
  overflow-x: auto;
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 7px;
  font-size: 0.85rem;
  line-height: 1.6;
}
.content pre code { background: none; border: 0; padding: 0; font-size: inherit; }

.table-wrap { overflow-x: auto; margin: 0 0 1.4rem; }
.content table { border-collapse: collapse; width: 100%; font-size: 0.87rem; }
.content th, .content td {
  padding: 0.55rem 0.8rem;
  border: 1px solid var(--border);
  text-align: left;
  vertical-align: top;
}
.content th { background: var(--surface-alt); font-weight: 620; }
.content td code { white-space: nowrap; }

/* landing cards, injected into the overview page */
.cards { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); margin: 0 0 2.5rem; }
.card {
  display: block;
  padding: 1.1rem 1.25rem;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface);
  color: var(--text);
}
.card:hover { border-color: var(--accent); text-decoration: none; }
.card-title { font-weight: 620; margin-bottom: 0.3rem; font-size: 0.98rem; }
.card-blurb { font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; }

.footer {
  margin-top: 4rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border);
  font-size: 0.83rem;
  color: var(--text-muted);
}

/* ---- responsive ---- */

@media (max-width: 60rem) {
  .layout { flex-direction: column; }
  .sidebar {
    position: static;
    width: 100%;
    flex-basis: auto;
    height: auto;
    border-right: 0;
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 1.5rem 1rem;
  }
  .toc-section { display: none; }
  .content { padding: 2rem 1.5rem 4rem; }
  .content h1 { font-size: 1.7rem; }
}
"""

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{blurb}">
<meta name="color-scheme" content="light dark">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='13' font-size='14'>&#x2744;</text></svg>">
<style>{css}</style>
</head>
<body>
<div class="layout">
<nav class="sidebar">
  <a class="brand" href="index.html">SnowBallSLR</a>
  <p class="brand-sub">Saturation-aware citation searching for systematic reviews</p>
  <div class="nav-label">Documentation</div>
  <ul class="nav">{nav}</ul>
  {toc_section}
  <div class="nav-label">Project</div>
  <ul class="nav">
    <li><a href="https://github.com/s-matysik/SnowBallSLR">Repository</a></li>
    <li><a href="https://github.com/s-matysik/SnowBallSLR/releases/tag/v1.0.0">Release v1.0.0</a></li>
    <li><a href="https://github.com/s-matysik/SnowBallSLR/issues">Issues</a></li>
  </ul>
</nav>
<main class="content">
{cards}
{body}
<div class="footer">
  <p>SnowBallSLR v1.0.0 &middot; MIT licence &middot;
  built from <code>{source}</code> in the
  <a href="https://github.com/s-matysik/SnowBallSLR">repository</a>.</p>
</div>
</main>
</div>
</body>
</html>
"""


def build_nav(current: Page) -> str:
    items = []
    for page in PAGES:
        cls = ' class="current"' if page.output == current.output else ""
        items.append(f'<li><a href="{page.output}"{cls}>{html.escape(page.title)}</a></li>')
    return "".join(items)


def build_toc(tokens: list[dict]) -> str:
    """A flat list of the level-2 headings -- deeper nesting is noise at this size.

    `toc_tokens` is a tree: the H2s hang off the document's H1, so collect
    recursively rather than reading the top level only.
    """

    def walk(nodes: list[dict]) -> list[dict]:
        out: list[dict] = []
        for node in nodes:
            out.append(node)
            out.extend(walk(node.get("children", [])))
        return out

    entries = [t for t in walk(tokens) if t.get("level") == 2]
    if len(entries) < 2:
        return ""
    items = "".join(
        f'<li><a href="#{t["id"]}">{html.escape(t["name"])}</a></li>' for t in entries
    )
    return f'<div class="toc-section"><div class="nav-label">On this page</div><ul class="toc">{items}</ul></div>'


def build_cards(current: Page) -> str:
    """Navigation cards, shown only on the landing page."""
    if current.output != "index.html":
        return ""
    cards = "".join(
        f'<a class="card" href="{p.output}">'
        f'<div class="card-title">{html.escape(p.title)}</div>'
        f'<div class="card-blurb">{html.escape(p.blurb)}</div></a>'
        for p in PAGES
        if p.output != "index.html"
    )
    return f'<div class="cards">{cards}</div>'


def wrap_tables(body: str) -> str:
    """Wide tables scroll inside their own box rather than the page."""
    return body.replace("<table>", '<div class="table-wrap"><table>').replace(
        "</table>", "</table></div>"
    )


def render(page: Page, out_dir: Path) -> None:
    text = (ROOT / page.source).read_text(encoding="utf-8")

    md = markdown.Markdown(
        extensions=["extra", "toc", "sane_lists", "smarty"],
        extension_configs={"toc": {"permalink": False}},
    )
    body = wrap_tables(md.convert(text))

    # The document's own H1 doubles as the browser title.
    match = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.DOTALL)
    heading = re.sub(r"<[^>]+>", "", match.group(1)).strip() if match else page.title
    title = heading if page.output == "index.html" else f"{heading} — SnowBallSLR"

    out_dir.joinpath(page.output).write_text(
        TEMPLATE.format(
            title=html.escape(title),
            blurb=html.escape(page.blurb),
            css=CSS,
            nav=build_nav(page),
            toc_section=build_toc(md.toc_tokens),
            cards=build_cards(page),
            body=body,
            source=html.escape(page.source),
        ),
        encoding="utf-8",
    )
    print(f"  {page.source:<28} -> {page.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="_site", help="output directory (default: _site)")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"Building site into {out_dir}")
    for page in PAGES:
        render(page, out_dir)

    figures = ROOT / "figures"
    if figures.is_dir():
        shutil.copytree(figures, out_dir / "figures")
        print(f"  figures/                     -> figures/")

    # Tell GitHub Pages not to run the output through Jekyll.
    out_dir.joinpath(".nojekyll").write_text("", encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
