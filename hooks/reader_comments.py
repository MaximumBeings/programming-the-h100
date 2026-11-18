"""Apply literal comment-only reader corrections; preserve executed source files."""
import re

CLUSTER_PAGE = '02-clusters-inline-ptx.md'
CLUSTER_OLD = '// Block 0 writes 0.0, Block 1 writes 100.0, Block 2 writes 200.0, ...'
CLUSTER_NEW = '// Example values by block: 0.0, 100.0, 200.0 (and so on).'

PAGE = '11-multi-gpu-nccl.md'
EXAMPLE = 'ex13_hybrid_parallel_training.cu'
OLD = '''    // Check GPU count
    // (See ex12's comment on the same pattern: initialize explicitly,
    // since cudaGetDeviceCount's return code isn't checked here and
    // an uninitialized n_gpus can read back as garbage -- reproduced
    // live as "found -1" when this exact uninitialized version ran
    // with zero CUDA-capable devices present.)
'''
NEW = '    // Check the visible device count before creating the eight-GPU layout.\n' + '\n' * 5
FENCES = re.compile(r'^```[^\n]*\n(.*?)^```', re.M | re.S)


def display_source(source):
    """Literal, comment-only correction; line numbers and all executable bytes stay exact."""
    if source.count(OLD) != 1:
        raise ValueError('Expected exact obsolete ex13 comment once; review the literal registry.')
    if 'CHECK_CUDA(cudaGetDeviceCount(&n_gpus));' not in source:
        raise ValueError('Expected checked device query in executed ex13 source.')
    if OLD.count('\n') != NEW.count('\n'):
        raise ValueError('Comment correction must preserve line numbers.')
    if any(line.strip() and not line.lstrip().startswith('//') for line in (OLD + NEW).splitlines()):
        raise ValueError('Reader correction may only replace whole comment lines.')
    result = source.replace(OLD, NEW, 1)
    if source.replace(OLD, '', 1) != result.replace(NEW, '', 1):
        raise ValueError('Reader correction changed material outside the exact comment.')
    return result


def on_page_markdown(markdown, page, **kwargs):
    if page.file.src_path == CLUSTER_PAGE:
        if markdown.count(CLUSTER_OLD) != 1:
            raise ValueError('Expected the exact cluster-value comment once.')
        return markdown.replace(CLUSTER_OLD, CLUSTER_NEW, 1)
    if page.file.src_path != PAGE:
        return markdown
    found = []
    for match in FENCES.finditer(markdown):
        source = match[1]
        if source.lstrip().startswith('//') and EXAMPLE in source and re.search(r'\bint\s+main\s*\(', source):
            found.append(match)
    if len(found) != 1:
        raise ValueError('Expected exactly one complete ex13 listing for reader correction.')
    match = found[0]
    return markdown[:match.start(1)] + display_source(match[1]) + markdown[match.end(1):]
