"""How each auditor verdict is shown: colour, wording, and the escaped HTML.

Kept out of `app.py` so it can be tested: Streamlit runs the app at import, so
nothing inside it is reachable from the offline suite, and this is exactly the
code where a wrong label or a missing escape would mislead a reader.

The wording is measured, not decorative (PLAN.md E4, RAGTruth QA test):
- green is right 98.5% of the time, so it may say "Verified";
- orange flags sentences that are 3.2x likelier than average to be unsupported,
  yet ~79% of flagged sentences are still supported - so it says "Low support -
  check it", never "unsupported" or "hallucinated";
- there is no red. It measured 7.7% precision against a 6.7% base rate (D7).
"""
import html

from verify.nli_checker import VERDICTS

VERDICT_STYLES = {
    'ENTAILMENT': {
        'css': 'sentence-verified',
        'icon': '✅',
        'label': 'Verified',
        'color': '#10b981',
        'rgb': '16, 185, 129',
        'lead': 'Supported by',
    },
    'NEUTRAL': {
        'css': 'sentence-neutral',
        'icon': '⚠️',
        'label': 'Not verified',
        'color': '#fbbf24',
        'rgb': '251, 191, 36',
        # For NEUTRAL the winning chunk is the closest passage, NOT support;
        # calling it "supporting" would assert exactly what NEUTRAL denies.
        'lead': 'Closest passage (does not establish the claim)',
    },
    'LOW_SUPPORT': {
        'css': 'sentence-low-support',
        'icon': '🔶',
        'label': 'Low support - check it',
        'color': '#f97316',
        'rgb': '249, 115, 22',
        'lead': 'Closest passage (supports this far less than usual - check it)',
    },
}

if set(VERDICT_STYLES) != set(VERDICTS):
    raise RuntimeError(
        f"ui.verdicts styles {sorted(VERDICT_STYLES)} do not match the auditor's "
        f"verdicts {sorted(VERDICTS)}; a verdict would render unstyled or not at all."
    )


def style(verdict):
    try:
        return VERDICT_STYLES[verdict]
    except KeyError:
        raise ValueError(f"no display style for verdict {verdict!r}; "
                         f"expected one of {VERDICTS}") from None


def format_entailment(p):
    """P(entailment) for display. The floor is ~0.0006, so small values keep
    their magnitude instead of rounding to a misleading 0.00."""
    if p is None:
        return 'n/a'
    if p >= 0.01:
        return f"{p:.2f}"
    if p > 0:
        return f"{p:.1e}"
    return '0'


def tooltip(result):
    s = style(result['verdict'])
    text = f"{s['icon']} {s['label']} | P(entailment) {format_entailment(result.get('entailment'))}"
    source = (result.get('evidence') or {}).get('chunk_id')
    if source:
        text += f" | {source}"
    return text


def response_html(results):
    """The answer with one underlined span per sentence. Everything from a
    document or the generator is escaped: both reach an HTML body/attribute."""
    spans = [
        f'<span class="{style(r["verdict"])["css"]}" '
        f'data-tooltip="{html.escape(tooltip(r), quote=True)}">'
        f'{html.escape(r["sentence"])}</span>'
        for r in results
    ]
    return '<div class="response-container">' + ' '.join(spans) + '</div>'


def counts(results):
    tally = {verdict: 0 for verdict in VERDICTS}
    for r in results:
        tally[r['verdict']] += 1
    return tally


def stats_html(results):
    tally = counts(results)
    badges = [
        f'<div class="stat-badge"><span class="stat-value" style="color: {style(v)["color"]};">'
        f'{tally[v]}</span><span class="stat-label">{style(v)["icon"]} {style(v)["label"]}</span></div>'
        for v in VERDICTS
    ]
    badges.append(f'<div class="stat-badge"><span class="stat-value">{len(results)}</span>'
                  f'<span class="stat-label"> Total Claims</span></div>')
    return '<div class="stats-container">' + ''.join(badges) + '</div>'


def legend_html():
    items = ''.join(
        f'<div class="legend-item"><div class="legend-line" '
        f'style="background: {style(v)["color"]};"></div><span>{style(v)["label"]}</span></div>'
        for v in VERDICTS
    )
    return f'<div class="legend-container">{items}</div>'


def underline_css():
    """Underline and tooltip rules for every verdict, from the one colour table."""
    rules = []
    for verdict in VERDICTS:
        s = style(verdict)
        rules.append(
            f".{s['css']} {{ position: relative; cursor: help; "
            f"border-bottom: 3px solid rgba({s['rgb']}, 0.6); "
            f"transition: all 0.2s ease; padding-bottom: 2px; }}\n"
            f".{s['css']}:hover {{ background: rgba({s['rgb']}, 0.1); "
            f"border-bottom-color: {s['color']}; }}"
        )
    classes = [f".{style(v)['css']}" for v in VERDICTS]
    rules.append(
        ',\n'.join(f"{c}::after" for c in classes) + " {\n"
        "    content: attr(data-tooltip); position: absolute; bottom: 100%; left: 50%;\n"
        "    transform: translateX(-50%) translateY(-8px); background: rgba(0, 0, 0, 0.95);\n"
        "    color: white; padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;\n"
        "    white-space: nowrap; opacity: 0; pointer-events: none;\n"
        "    transition: opacity 0.2s ease, transform 0.2s ease; z-index: 1000;\n"
        "    font-family: 'JetBrains Mono', monospace;\n}"
    )
    rules.append(
        ',\n'.join(f"{c}:hover::after" for c in classes)
        + " {\n    opacity: 1; transform: translateX(-50%) translateY(-12px);\n}"
    )
    return '\n'.join(rules)


def analysis_markdown(index, result):
    """One Detailed Analysis entry. The sentence is escaped: this block is
    rendered with unsafe_allow_html, and it previously was not."""
    s = style(result['verdict'])
    return (
        f"**{index}. {s['icon']} {html.escape(result['sentence'])}**\n"
        f"- Verdict: <span style=\"color: {s['color']}; font-weight: 600;\">{s['label']}</span>\n"
        f"- P(entailment): {format_entailment(result.get('entailment'))}"
    )
