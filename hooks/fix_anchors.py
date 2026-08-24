"""
MkDocs hook: repair the hand-written in-page "Table of Contents" links.

The source chapters were written with abbreviated anchors, e.g. a link to
`#5--connectx-7` pointing at a heading whose real slug is
`5--connectx-7-and-the-inter-node-bottleneck`. Rather than editing the
original markdown, this hook rewrites those links at build time by
prefix-matching each broken anchor against the page's real heading slugs.

Source files stay byte-for-byte untouched on disk.
"""

from __future__ import annotations

import re

from markdown.extensions.toc import slugify

# Chapters that were split across two files: an anchor the first file's table
# of contents cannot resolve is looked up in its continuation instead, and
# rewritten as a cross-page link.
CONTINUATIONS = {
    "08-kernel-design-part-1.md": "09-kernel-design-part-2.md",
}

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
LINK_RE = re.compile(r"\]\(#([^)\s]+)\)")
FENCE_RE = re.compile(r"^(?:```|~~~)", re.MULTILINE)


def _plain(text: str) -> str:
    """Approximate the plain-text form of a heading the way python-markdown does."""
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)  # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links
    text = text.replace("`", "")
    text = re.sub(r"\*\*|__|\*|~~", "", text)
    return text.strip()


def _normalize(slug: str) -> str:
    return re.sub(r"-+", "-", slug.lower()).strip("-")


def _heading_slugs(markdown: str) -> list[str]:
    """Real heading slugs for the page, in document order, skipping fenced code."""
    slugs: list[str] = []
    in_fence = False
    for line in markdown.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING_RE.match(line)
        if m:
            slugs.append(slugify(_plain(m.group(2)), "-"))
    return slugs


def _lookup(slugs: list[str], anchor: str) -> str | None:
    """Best match for `anchor` among `slugs`, or None."""
    norm_map: dict[str, str] = {}
    for s in slugs:
        norm_map.setdefault(_normalize(s), s)
    key = _normalize(anchor)
    if key in norm_map:
        return norm_map[key]
    candidates = [s for n, s in norm_map.items() if n.startswith(key)]
    if not candidates:
        head = key.split("-")[0]
        if head.isdigit():
            candidates = [s for n, s in norm_map.items() if n.startswith(f"{head}-")]
    return min(candidates, key=len) if candidates else None


def _continuation_slugs(page, files) -> tuple[str, list[str]] | None:
    target = CONTINUATIONS.get(page.file.src_uri)
    if not target:
        return None
    other = files.get_file_from_path(target)
    if other is None or not other.abs_src_path:
        return None
    try:
        with open(other.abs_src_path, encoding="utf-8") as fh:
            return target, _heading_slugs(fh.read())
    except OSError:
        return None


def on_page_markdown(markdown, page, config, files, **kwargs):
    # Chapters split mid-sequence have no H1 of their own; give them the title
    # the nav uses so every chapter page opens the same way.
    if not re.search(r"^# [^#]", markdown, re.MULTILINE):
        title = getattr(page, "title", None)
        if title:
            body = markdown.lstrip("- \t\r\n")
            markdown = "# " + title + "\n\n" + body

    slugs = _heading_slugs(markdown)
    if not slugs:
        return markdown

    continuation = _continuation_slugs(page, files)
    known = set(slugs)
    # The source anchors were written GitHub-style, where an em dash becomes a
    # hyphen ("#1--the-big-picture"); python-markdown drops it instead
    # ("#1-the-big-picture"). _lookup compares with runs of hyphens collapsed so
    # the two conventions line up.
    cache: dict[str, str | None] = {}

    def resolve(anchor: str) -> str | None:
        if anchor in cache:
            return cache[anchor]
        result: str | None = None
        if anchor not in known:
            hit = _lookup(slugs, anchor)
            if hit:
                result = f"#{hit}"
            elif continuation:
                target, other_slugs = continuation
                hit = _lookup(other_slugs, anchor)
                if hit:
                    result = f"{target}#{hit}"
        cache[anchor] = result
        return result

    def repl(match: re.Match) -> str:
        target = resolve(match.group(1))
        return f"]({target})" if target else match.group(0)

    return LINK_RE.sub(repl, markdown)
