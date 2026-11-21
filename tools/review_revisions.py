"""Verify exact pending source transitions; check separately captured corrected GPU runs.

Historical verifiers use the returned baseline identities only after verifying
both the current revision and its immutable pre-correction source. Raw evidence
checks remain unchanged. The separate reader notice is mandatory.
"""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revisions(root):
    rows = json.loads((root/'tools/review_source_revisions.json').read_text())
    from validate_examples import programs
    current = {(p['page'], p['example']): p['sha256'] for p in programs()}
    seen = set()
    for row in rows:
        key = (row['page'], row['example'])
        if key in seen or row['status'] not in {'pending_gpu_revalidation','verified_gpu_revalidation'}:
            raise ValueError('Invalid pending review revision: '+str(key))
        seen.add(key)
        for field in ('archived_source', 'page', 'source'):
            if field in row:
                rel = Path(row[field])
                if rel.is_absolute() or '..' in rel.parts:
                    raise ValueError('Unsafe correction path')
        before, after = row['archived_sha256'], row['current_sha256']
        if before == after or digest(root/row['archived_source']) != before:
            raise ValueError('Changed pre-correction snapshot: '+str(key))
        actual = digest(root/row['source']) if 'source' in row else current.get(key)
        if actual != after:
            raise ValueError('Unregistered current source drift: '+str(key))
        text = (root/row['page']).read_text()
        notice = 'GPU revalidation completed' if row['status'] == 'verified_gpu_revalidation' else 'GPU revalidation pending'
        if notice not in text or 'review-fixes-2026-11-21.md' not in text:
            raise ValueError('Missing visible pending notice: '+str(key))
        report = (root/'docs/review-fixes-2026-11-21.md').read_text()
        if before not in report or after not in report:
            raise ValueError('Pending identity absent from public repair report')
    if any(r['status'] == 'verified_gpu_revalidation' for r in rows):
        from verify_current_h100 import verified_sources
        captured = verified_sources(root)
        for row in rows:
            if row['status'] == 'verified_gpu_revalidation' and captured.get((row['page'],row['example'])) != row['current_sha256']:
                raise ValueError('Corrected source lacks matching successful rerun')
    return rows


def historical_hashes(root, current):
    """Return pinned PRE-correction hashes for historical-evidence checks."""
    result = dict(current)
    for row in revisions(root):
        key = (row['page'], row['example'])
        if 'source' not in row:
            if current.get(key) != row['current_sha256']:
                raise ValueError('Historical caller supplied a different source')
            result[key] = row['archived_sha256']
    return result


def historical_file_hash(root, source):
    for row in revisions(root):
        if row.get('source') == source:
            return row['archived_sha256']
    return digest(root/source)
