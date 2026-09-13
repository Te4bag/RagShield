"""Verdict display: every verdict styled, nothing unescaped, no red, honest words.

Streamlit executes `app.py` at import, so this module is the testable surface
of the UI. What is pinned: the style table matches the auditor's verdicts
exactly, document text can never become markup, the low-support label never
claims more than E4 measured, and small P(entailment) values stay legible.
"""
import re

import pytest

import ui
from verify.nli_checker import VERDICTS


def result(verdict, sentence="A sentence.", entailment=0.5, chunk_id="doc.pdf_ch0"):
    return {'sentence': sentence, 'verdict': verdict, 'entailment': entailment,
            'probabilities': None,
            'evidence': {'chunk_id': chunk_id, 'doc_id': 'doc.pdf', 'text': 'passage'}}


def test_every_auditor_verdict_has_a_style_and_nothing_else_does():
    assert set(ui.VERDICT_STYLES) == set(VERDICTS)


def test_an_unknown_verdict_fails_loudly_instead_of_rendering_unstyled():
    with pytest.raises(ValueError, match="no display style"):
        ui.style('CONTRADICTION')


def test_there_is_no_red_anywhere():
    """D7 removed red; no colour, label or class may bring it back."""
    rendered = ' '.join([ui.underline_css(), ui.legend_html(),
                         ui.stats_html([result(v) for v in VERDICTS])]).lower()

    assert '#ef4444' not in rendered
    assert 'contradict' not in rendered


def test_low_support_wording_never_overstates_the_measurement():
    """~79% of flagged sentences are supported (E4), so the flag must not say
    unsupported, hallucinated, false or wrong."""
    s = ui.style('LOW_SUPPORT')
    words = f"{s['label']} {s['lead']}".lower()

    for claim in ('unsupported', 'hallucinat', 'false', 'wrong', 'contradict'):
        assert claim not in words
    assert 'check' in words


def test_neutral_evidence_is_not_called_support():
    assert 'support' not in ui.style('NEUTRAL')['lead'].lower().replace('does not establish', '')
    assert ui.style('ENTAILMENT')['lead'] == 'Supported by'


def test_response_html_escapes_sentences_and_tooltips():
    """A document containing markup must not render as markup."""
    nasty = result('NEUTRAL', sentence='<b>bold</b> & "quoted"',
                   chunk_id='x" onmouseover="alert(1)')

    out = ui.response_html([nasty])

    assert '<b>' not in out
    assert '&lt;b&gt;bold&lt;/b&gt; &amp;' in out
    assert 'onmouseover="alert' not in out
    assert '&quot;' in out


def test_response_html_uses_one_span_per_sentence_with_its_verdict_class():
    results = [result('ENTAILMENT', 'One.'), result('LOW_SUPPORT', 'Two.'),
               result('NEUTRAL', 'Three.')]

    out = ui.response_html(results)

    classes = re.findall(r'<span class="([^"]+)"', out)
    assert classes == ['sentence-verified', 'sentence-low-support', 'sentence-neutral']


def test_detailed_analysis_escapes_the_sentence():
    """This block renders with unsafe_allow_html and was not escaped before P8."""
    out = ui.analysis_markdown(1, result('ENTAILMENT', sentence='<img src=x onerror=alert(1)>'))

    assert '<img' not in out
    assert '&lt;img' in out


@pytest.mark.parametrize("p, shown", [
    (None, 'n/a'),
    (0.97, '0.97'),
    (0.0104, '0.01'),
    (0.000552, '5.5e-04'),
    (0.0, '0'),
])
def test_entailment_keeps_its_magnitude_near_the_floor(p, shown):
    """The floor is 0.000552; rounding to 2 places would show every flagged
    sentence as 0.00 and hide how far below the floor it is."""
    assert ui.format_entailment(p) == shown


def test_tooltip_names_label_score_and_source():
    tip = ui.tooltip(result('LOW_SUPPORT', entailment=0.0003))

    assert ui.style('LOW_SUPPORT')['label'] in tip
    assert '3.0e-04' in tip
    assert 'doc.pdf_ch0' in tip


def test_tooltip_without_evidence_omits_the_source():
    r = result('NEUTRAL', entailment=None)
    r['evidence'] = None

    assert ui.tooltip(r).endswith('P(entailment) n/a')


def test_counts_cover_every_verdict_even_when_absent():
    tally = ui.counts([result('ENTAILMENT'), result('ENTAILMENT'), result('LOW_SUPPORT')])

    assert tally == {'ENTAILMENT': 2, 'NEUTRAL': 0, 'LOW_SUPPORT': 1}


def test_stats_and_legend_show_every_verdict_once():
    results = [result('ENTAILMENT'), result('LOW_SUPPORT')]

    stats = ui.stats_html(results)
    legend = ui.legend_html()

    for verdict in VERDICTS:
        label = ui.style(verdict)['label']
        assert stats.count(label) == 1
        assert legend.count(label) == 1
    assert '>2</span><span class="stat-label"> Total Claims' in stats


def test_css_defines_underline_hover_and_tooltip_for_every_class():
    css = ui.underline_css()

    for verdict in VERDICTS:
        cls = ui.style(verdict)['css']
        assert f".{cls} {{" in css
        assert f".{cls}:hover {{" in css
        assert f".{cls}::after" in css
        assert f".{cls}:hover::after" in css
