#!/usr/bin/env python3
"""Turn LaTeX math in an HTML source file into MathML.

    pip install latex2mathml
    python3 _tools/texmath.py _src/conformal_sphere.src.html conformal_sphere.html

In the source, $...$ is inline math and $$...$$ is display math (a trailing \\tag{n} becomes a right-aligned
equation number). Everything inside <pre>...</pre> is left alone. The output needs math.css for the font and
the equation layout.

latex2mathml's output is adjusted for Chrome's MathML Core, which has no mathvariant other than "normal":
single-letter upright symbols, bold vectors, brackets that must not stretch, and operators that the converter
emits as identifiers.
"""
import html
import re
import sys

import latex2mathml.converter as l2m

# --- macros (plain text substitution before conversion) -------------------------------------------------------
SIMPLE = {
    r"\dd": r"\text{d}",          # upright d of the exterior derivative
    r"\dA": r"\text{d}A",
    r"\R": r"\mathbb{R}",
    r"\wdg": r"\wedge",
    r"\comp": r"\circ",
}
ARGS2 = {
    r"\ip": r"\langle #1, #2\rangle",       # inner product <a, b>
    r"\wp": r"\langle #1 \wedge #2\rangle",  # <alpha ^ beta>
}


def expand_macros(t):
    for name, body in ARGS2.items():
        pat = re.escape(name) + r"\{((?:[^{}]|\{[^{}]*\})*)\}\{((?:[^{}]|\{[^{}]*\})*)\}"
        t = re.sub(pat, lambda m: body.replace("#1", m.group(1)).replace("#2", m.group(2)), t)
    for name in sorted(SIMPLE, key=len, reverse=True):          # \dA before \dd
        t = re.sub(re.escape(name) + r"(?![A-Za-z])", lambda m: SIMPLE[name], t)
    return t


# --- fixes to latex2mathml output ---------------------------------------------------------------------------------
def bold_char(c):
    if "a" <= c <= "z":
        return chr(0x1D41A + ord(c) - ord("a"))
    if "A" <= c <= "Z":
        return chr(0x1D400 + ord(c) - ord("A"))
    return c


def fix(mathml):
    m = mathml.replace(' xmlns="http://www.w3.org/1998/Math/MathML"', "")
    m = re.sub(r'<mi mathvariant="bold">(.)</mi>', lambda g: "<mi>%s</mi>" % bold_char(g.group(1)), m)
    m = m.replace("<mtext>d</mtext>", '<mi mathvariant="normal">d</mi>')
    m = re.sub(r"<mi>(&#x027E8;|&#x027E9;)</mi>", r'<mo stretchy="false">\1</mo>', m)        # angle brackets
    m = re.sub(r"<mi>([:;,])</mi>", r"<mo>\1</mo>", m)
    m = m.replace("<mi>cot</mi>", "<mi>cot</mi><mo>&#x02061;</mo>")                            # function application: thin space
    m = m.replace("<mo>&#x0002D;</mo>", "<mo>&#x02212;</mo>")                                   # hyphen -> minus
    # |x| bars and the Hodge star are used as tight symbols, not as spaced infix operators
    m = m.replace('<mo stretchy="false">&#x0007C;</mo>', '<mo stretchy="false" lspace="0" rspace="0">&#x0007C;</mo>')
    m = m.replace("<mo>&#x022C6;</mo>", '<mo lspace="0" rspace="0">&#x022C6;</mo>')
    # a minus right after another operator (=, +, an opening bracket, a comma) is a sign, not a subtraction;
    # MathML Core infers "infix" for it because it is not first in its row
    def sign(g):
        prev = g.group(2)
        closing = prev in ("&#x00029;", "&#x0005D;", "&#x0007C;", "&#x027E9;")
        return g.group(0) if closing else g.group(1) + '<mo form="prefix">&#x02212;</mo>'
    m = re.sub(r"(<mo[^>]*>(&#x[0-9A-Fa-f]+;)</mo>)<mo>&#x02212;</mo>", sign, m)
    return m


def convert(tex, display):
    tex = expand_macros(html.unescape(tex.strip()))
    return fix(l2m.convert(tex, display="block" if display else "inline"))


def process(text):
    def display_sub(m):
        body = m.group(1)
        tag = re.search(r"\\tag\{([^}]*)\}", body)
        body = re.sub(r"\\tag\{[^}]*\}", "", body)
        out = '<div class="eq"><div class="eqbody">%s</div>' % convert(body, True)
        if tag:
            out += '<span class="eqno">(%s)</span>' % tag.group(1)
        return out + "</div>"

    text = re.sub(r"\$\$(.+?)\$\$", display_sub, text, flags=re.S)
    text = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", lambda m: convert(m.group(1), False), text, flags=re.S)
    # keep punctuation after inline math on the same line (a word joiner forbids the break)
    text = re.sub(r"(</math>)([.,;:!?)])", "\\1&#8288;\\2", text)
    return text


def main(src, dst):
    s = open(src, encoding="utf-8").read()
    parts = re.split(r"(<pre>.*?</pre>)", s, flags=re.S)
    out = "".join(p if p.startswith("<pre>") else process(p) for p in parts)
    open(dst, "w", encoding="utf-8").write(out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
