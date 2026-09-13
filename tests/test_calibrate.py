"""E4 floor derivation and gate audit, on hand-sized fixtures.

The floor is the one number E4 writes into config.yaml, so its selection rule
is pinned exactly: largest floor within the train FPR target, ties never
pushing past it, rounding only ever down.
"""
import math

import numpy as np
import pytest

from eval import calibrate as cal


# ------------------------------------------------------------------- floor

# 10 supported sentences with P(entailment) .01 .02 ... .10, 2 unsupported.
SUPPORTED_P = [0.01 * i for i in range(1, 11)]
GOLD = [False] * 10 + [True, True]
P = SUPPORTED_P + [0.001, 0.5]


def test_floor_is_the_largest_value_within_the_target():
    """Target 20% of 10 supported -> k = 2 false positives allowed. Flagging
    P < .03 catches .01 and .02 (2 FPs); any larger floor catches .03 too."""
    floor = cal.floor_at_fpr(GOLD, P, 0.20)

    assert floor == pytest.approx(0.03)
    assert cal.operating_point(GOLD, P, floor)['fpr'] == pytest.approx(0.20)
    assert cal.operating_point(GOLD, P, floor + 1e-9)['fpr'] > 0.20


def test_zero_target_flags_nothing_supported():
    floor = cal.floor_at_fpr(GOLD, P, 0.0)

    assert floor == pytest.approx(0.01)
    assert cal.operating_point(GOLD, P, floor)['fpr'] == 0.0


def test_a_target_between_steps_rounds_down_to_the_allowed_count():
    """15% of 10 = 1.5 false positives: only 1 is allowed."""
    floor = cal.floor_at_fpr(GOLD, P, 0.15)

    assert cal.operating_point(GOLD, P, floor)['fpr'] == pytest.approx(0.10)


def test_ties_at_the_floor_never_push_past_the_target():
    """Supported scores .01 .02 .02 .02: 25% allows 1 FP. The next value is .02,
    shared by three sentences, so the floor must stop below all of them."""
    gold = [False] * 4 + [True]
    p = [0.01, 0.02, 0.02, 0.02, 0.0001]

    floor = cal.floor_at_fpr(gold, p, 0.25)

    assert floor == pytest.approx(0.02)
    assert cal.operating_point(gold, p, floor)['fpr'] == pytest.approx(0.25)


def test_a_target_of_one_allows_flagging_everything():
    assert cal.floor_at_fpr(GOLD, P, 1.0) == math.inf


def test_floor_needs_supported_sentences():
    with pytest.raises(ValueError, match="no supported sentences"):
        cal.floor_at_fpr([True, True], [0.1, 0.2], 0.05)


def test_target_outside_zero_one_is_rejected():
    with pytest.raises(ValueError, match="target_fpr"):
        cal.floor_at_fpr(GOLD, P, 5)


@pytest.mark.parametrize("value, expected", [
    (0.0012345, 0.00123),
    (0.0019999, 0.00199),
    (0.5, 0.5),
    (0.123456, 0.123),
])
def test_floor_rounds_down_to_three_significant_figures(value, expected):
    assert cal.round_floor_down(value) == pytest.approx(expected)
    assert cal.round_floor_down(value) <= value


def test_rounding_down_can_only_lower_the_fpr():
    rng = np.random.default_rng(0)
    gold = rng.random(400) < 0.2
    p = rng.random(400) ** 6            # piled up near zero, like the real head

    raw = cal.floor_at_fpr(gold, p, 0.05)

    assert (cal.operating_point(gold, p, cal.round_floor_down(raw))['fpr']
            <= cal.operating_point(gold, p, raw)['fpr'] <= 0.05)


def test_operating_point_counts():
    """Floor .03: flags .01 .02 (supported) and .001 (unsupported).
    TP 1, FP 2 -> precision 1/3, TPR 1/2, FPR 2/10."""
    point = cal.operating_point(GOLD, P, 0.03)

    assert point['flagged'] == 3
    assert point['precision'] == pytest.approx(1 / 3)
    assert point['tpr'] == pytest.approx(0.5)
    assert point['fpr'] == pytest.approx(0.2)
    assert point['base_rate'] == pytest.approx(2 / 12)


# -------------------------------------------------------------- gate audit

def s(p_ent, p_con, unsupported):
    return {'p_entailment': p_ent, 'p_contradiction': p_con,
            'p_neutral': 1.0 - p_ent - p_con, 'unsupported': unsupported}


def test_verdict_at_mirrors_the_auditor():
    assert cal.verdict_at(s(0.90, 0.05, False), 0.85, 0.85) == 'ENTAILMENT'
    assert cal.verdict_at(s(0.80, 0.05, False), 0.85, 0.85) == 'NEUTRAL'     # demoted
    assert cal.verdict_at(s(0.02, 0.95, True), 0.85, 0.85) == 'CONTRADICTION'
    assert cal.verdict_at(s(0.02, 0.60, True), 0.85, 0.85) == 'NEUTRAL'      # demoted
    assert cal.verdict_at(s(0.10, 0.10, True), 0.85, 0.85) == 'NEUTRAL'      # argmax neutral


SENTENCES = [
    s(0.99, 0.00, False),   # green, right
    s(0.92, 0.01, False),   # green at .85/.90, right
    s(0.88, 0.02, True),    # green at .85 only, wrong
    s(0.01, 0.97, True),    # red at .85/.95, right
    s(0.01, 0.90, False),   # red at .85/.90, wrong
    s(0.01, 0.87, False),   # red at .85 only, wrong
    s(0.30, 0.10, False),   # neutral
]


def test_green_audit_precision_and_reach():
    rows = {r['gate']: r for r in cal.gate_audit(SENTENCES, 'ENTAILMENT',
                                                 (0.85, 0.90), 0.85, 0.85)}

    # .85: greens are .99 .92 .88 -> 2 of 3 right; supported sentences total 5.
    assert rows[0.85]['n'] == 3
    assert rows[0.85]['precision'] == pytest.approx(2 / 3)
    assert rows[0.85]['share'] == pytest.approx(2 / 5)
    # .90: .99 .92 -> both right.
    assert rows[0.90]['precision'] == pytest.approx(1.0)


def test_red_audit_precision_rises_with_the_gate():
    rows = {r['gate']: r for r in cal.gate_audit(SENTENCES, 'CONTRADICTION',
                                                 (0.85, 0.90, 0.95), 0.85, 0.85)}

    assert (rows[0.85]['n'], rows[0.85]['precision']) == (3, pytest.approx(1 / 3))
    assert (rows[0.90]['n'], rows[0.90]['precision']) == (2, pytest.approx(1 / 2))
    assert (rows[0.95]['n'], rows[0.95]['precision']) == (1, pytest.approx(1.0))
    assert rows[0.95]['share'] == pytest.approx(1 / 2)     # 2 unsupported in total


# ------------------------------------------------------------- run pairing

def meta(**changes):
    m = {'config': {'nli_model': 'x', 'ingestion': {'chunk_size': 600, 'chunk_overlap': 60},
                    'verification': {'max_length': 512, 'entailment_threshold': 0.85}},
         'segmenter': {'spacy': '3.8.15'}}
    for path, value in changes.items():
        node = m
        *parents, leaf = path.split('.')
        for p in parents:
            node = node[p]
        node[leaf] = value
    return m


def test_runs_differing_only_in_thresholds_can_be_paired():
    """E4 adds a threshold to config; runs scored before it must still pair."""
    assert cal.scoring_mismatch(meta(), meta(**{'config.verification.entailment_threshold': 0.5})) == ''


@pytest.mark.parametrize("path, value", [
    ('config.nli_model', 'other'),
    ('config.ingestion.chunk_size', 300),
    ('config.verification.max_length', 256),
    ('segmenter', {'spacy': '4.0'}),
])
def test_runs_scored_differently_cannot_be_paired(path, value):
    assert cal.scoring_mismatch(meta(), meta(**{path: value}))


# ------------------------------------------------------------- reliability

def test_reliability_svg_is_well_formed_and_escapes_the_title():
    import xml.etree.ElementTree as ET
    from eval import metrics

    c = metrics.calibration([0, 1, 1, 0], [0.1, 0.8, 0.9, 0.3], n_bins=4)

    svg = cal.reliability_svg(c, 'QA <train> & test')

    root = ET.fromstring(svg)
    assert root.tag.endswith('svg')
    assert 'QA &lt;train&gt; &amp; test' in svg
    assert svg.count('<circle') == sum(1 for b in c.bins if b.count)
