"""Validate explicitly pinned post-execution corrections without claiming a rerun."""
import json
import re


def pending(root, current, recorded):
    path = root / 'tools/pending_gpu_revalidation.json'
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    result = {}
    for row in rows:
        key = (row['page'], row['example'])
        if key in result or row['status'] != 'pending_gpu_revalidation' or not row['reason'].strip():
            raise ValueError('Invalid/duplicate pending GPU correction')
        before, after = row['archived_sha256'], row['current_sha256']
        if before == after or not all(re.fullmatch(r'[0-9a-f]{64}', value) for value in (before, after)):
            raise ValueError('Invalid pending source hashes: ' + str(key))
        if recorded.get(key) != before or current.get(key) != after:
            raise ValueError('Pending correction differs from pinned revisions: ' + str(key))
        if 'GPU revalidation pending' not in (root / row['page']).read_text():
            raise ValueError('Pending correction lacks a visible reader notice: ' + str(key))
        result[key] = row
    return result
