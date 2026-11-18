"""Format displayed C/C++/CUDA listings; reject changes to lexical tokens."""
from functools import lru_cache
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FENCES = re.compile(r'^(`{3,})([^\n]*)\n(.*?)^\1[ \t]*$', re.M | re.S)
LANGUAGES = {'c': 'example.c', 'cpp': 'example.cpp', 'c++': 'example.cpp', 'cuda': 'example.cu'}
# Translation phase 2 removes escaped newlines before tokenization. Preserve
# string/character contents and multi-character operators, not just characters.
LEXER = re.compile(
    r'(?P<comment>//[^\n]*|/\*.*?\*/)'
    r'|(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delimiter)"'
    r'|(?:u8|u|U|L)?"(?:\\.|[^"\\])*"'
    r"|(?:u8|u|U|L)?'(?:\\.|[^'\\])*'"
    r'|[A-Za-z_][A-Za-z_0-9]*'
    r'|(?:\d[\w.\']*|\.\d[\w.\']*)(?:[eEpP][+-]?[\w.\']+)?'
    r'|<<<|>>>|<<=|>>=|<=>|->\*|\.\.\.|::|##|\+\+|--|->|\.\*|<<|>>|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|/=|%=|&=|\|=|\^='
    r'|[^\s]', re.S)


def lexical_signature(source):
    source = re.sub(r'\\\r?\n', '', source)
    tokens, comments = [], []
    for match in LEXER.finditer(source):
        if match.lastgroup == 'comment':
            comments.append(' '.join(match[0].split()))
        else:
            tokens.append(match[0])
    return tokens, comments


@lru_cache(maxsize=1)
def executable():
    path = shutil.which('clang-format')
    sibling = Path(sys.executable).parent / 'clang-format'
    if not path and sibling.is_file():
        path = str(sibling)
    if not path:
        raise RuntimeError('Install the pinned requirements before building: clang-format is missing.')
    version = subprocess.run([path, '--version'], check=True, capture_output=True, text=True).stdout
    if 'version 19.1.7' not in version:
        raise RuntimeError('Code presentation requires clang-format 19.1.7, as pinned in requirements.txt.')
    return path


@lru_cache(maxsize=1024)
def format_source(source, language):
    signature = lexical_signature(source)
    result = source
    # A wrapped trailing comment can settle on the next pass. Require a fixed
    # point, so rebuilding or copying a formatted listing never churns it.
    for _ in range(4):
        previous = result
        result = subprocess.run(
            [executable(), '--style=file:' + str(ROOT / '.clang-format'),
             '--assume-filename=' + LANGUAGES[language]],
            input=previous, text=True, capture_output=True, check=True, timeout=15).stdout
        if lexical_signature(result) != signature:
            raise ValueError('Code presentation changed tokens/comments; review the formatter configuration.')
        if result == previous:
            return result
    raise ValueError('Code presentation did not reach a stable layout.')



def on_page_markdown(markdown, **kwargs):
    def replace(match):
        language = match[2].strip()
        if language not in LANGUAGES:
            return match[0]
        return match[1] + match[2] + '\n' + format_source(match[3], language) + match[1]
    return FENCES.sub(replace, markdown)
