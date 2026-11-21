"""Label immutable historical summaries without rewriting evidence files."""
HISTORICAL = {'revalidation-2026-11-18.md','final-audit.md','eight-gpu-validation.md',
              'cutlass-workshop-validation.md'}

def on_page_markdown(markdown, page, **kwargs):
    if page.file.src_path not in HISTORICAL:
        return markdown
    return ( '> **Historical execution record:** this page preserves its dated source and results. '
             'Subsequent corrections and GPU revalidation still pending are listed in the '
             '[November 21 repair record](review-fixes-2026-11-21.md). '
             'Offline hash verification does not execute the corrected programs.\n\n'+markdown)
