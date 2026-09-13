"""Metrics, checked against values computed by hand rather than by a library.

Every expected number in this file is worked out in the comment above it, on a
fixture small enough to do on paper. That is deliberate: sklearn is installed in
this environment as a transitive dependency, so asserting against it would look
like verification while really only asserting that two implementations agree --
and it would put a heavyweight import in an offline suite that does not need it.

The load-bearing test is `test_pooled_beats_within_when_scores_only_read_the_response`:
it builds the exact confound the module exists to expose, a scorer that knows
nothing about individual sentences and is rewarded for it by pooled AUROC.
"""
import numpy as np
import pytest

from eval import metrics as m

ENTAIL, NEUTRAL, CONTRA = 'ENTAILMENT', 'NEUTRAL', 'CONTRADICTION'
LABELS = (ENTAIL, NEUTRAL, CONTRA)

# 8 sentences. Gold / predicted, lined up:
#   gold  E E E E N N C C
#   pred  E E E N N N C E
GOLD = [ENTAIL, ENTAIL, ENTAIL, ENTAIL, NEUTRAL, NEUTRAL, CONTRA, CONTRA]
PRED = [ENTAIL, ENTAIL, ENTAIL, NEUTRAL, NEUTRAL, NEUTRAL, CONTRA, ENTAIL]


# ------------------------------------------------------------ verdict half

def test_confusion_matrix_counts():
    """Rows are gold, columns predicted:
           E  N  C
        E [3, 1, 0]
        N [0, 2, 0]
        C [1, 0, 1]
    """
    cm = m.confusion_matrix(GOLD, PRED, labels=LABELS)

    assert cm.labels == LABELS
    assert cm.counts.tolist() == [[3, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_confusion_matrix_infers_sorted_labels():
    cm = m.confusion_matrix(['b', 'a'], ['a', 'a'])

    assert cm.labels == ('a', 'b')
    assert cm.counts.tolist() == [[1, 0], [1, 0]]


def test_an_unobserved_label_still_gets_a_row():
    """Pass `labels` explicitly and a class absent from the run stays visible."""
    cm = m.confusion_matrix([ENTAIL], [ENTAIL], labels=LABELS)

    assert cm.counts.tolist() == [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    # And it must not drag balanced accuracy down: no gold, no recall.
    assert m.balanced_accuracy(cm) == 1.0


def test_confusion_matrix_columns_line_up_under_long_labels():
    """CONTRADICTION is 13 characters and the counts are one or two. Sizing the
    columns off the counts alone ran the header together and left it out of
    step with the rows underneath."""
    header, *rows = m.confusion_matrix(GOLD, PRED, labels=LABELS).to_text().splitlines()[1:]

    assert all(len(row) == len(header) for row in rows)
    for label in LABELS:
        column_end = header.index(label) + len(label)
        assert all(row[column_end - 1].isdigit() for row in rows)


def test_a_label_outside_the_stated_set_raises():
    with pytest.raises(ValueError, match="absent from"):
        m.confusion_matrix(['SOMETHING_ELSE'], [ENTAIL], labels=LABELS)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        m.confusion_matrix([ENTAIL, ENTAIL], [ENTAIL])


def test_per_class_precision_recall_f1():
    """By hand, from the matrix above:
        E: tp 3, pred 3+0+1=4, gold 4  -> P .75,   R .75, F1 .75
        N: tp 2, pred 1+2+0=3, gold 2  -> P 2/3,   R 1.0, F1 .8
        C: tp 1, pred 0+0+1=1, gold 2  -> P 1.0,   R .5,  F1 2/3
    """
    per_class = {c.label: c for c in
                 m.per_class_metrics(m.confusion_matrix(GOLD, PRED, labels=LABELS))}

    assert per_class[ENTAIL].precision == pytest.approx(0.75)
    assert per_class[ENTAIL].recall == pytest.approx(0.75)
    assert per_class[ENTAIL].f1 == pytest.approx(0.75)

    assert per_class[NEUTRAL].precision == pytest.approx(2 / 3)
    assert per_class[NEUTRAL].recall == pytest.approx(1.0)
    assert per_class[NEUTRAL].f1 == pytest.approx(0.8)

    assert per_class[CONTRA].precision == pytest.approx(1.0)
    assert per_class[CONTRA].recall == pytest.approx(0.5)
    assert per_class[CONTRA].f1 == pytest.approx(2 / 3)

    assert per_class[CONTRA].support == 2
    assert per_class[CONTRA].predicted == 1


def test_accuracy_balanced_accuracy_and_macro_f1():
    """accuracy (3+2+1)/8 = .75; balanced (.75+1+.5)/3 = .75;
       macro F1 (.75+.8+2/3)/3."""
    report = m.verdict_report(GOLD, PRED, labels=LABELS)

    assert report.n == 8
    assert report.accuracy == pytest.approx(0.75)
    assert report.balanced_accuracy == pytest.approx(0.75)
    assert report.macro_f1 == pytest.approx((0.75 + 0.8 + 2 / 3) / 3)


def test_balanced_accuracy_differs_from_accuracy_on_an_imbalanced_set():
    """The reason both are reported: 9 easy ENTAILMENTs hide a missed red."""
    gold = [ENTAIL] * 9 + [CONTRA]
    pred = [ENTAIL] * 10

    report = m.verdict_report(gold, pred, labels=LABELS)

    assert report.accuracy == pytest.approx(0.9)
    assert report.balanced_accuracy == pytest.approx(0.5)


def test_a_class_that_is_never_predicted_scores_zero_not_nan():
    cm = m.confusion_matrix([CONTRA], [NEUTRAL], labels=LABELS)
    per_class = {c.label: c for c in m.per_class_metrics(cm)}

    assert per_class[CONTRA].precision == 0.0     # never predicted
    assert per_class[CONTRA].recall == 0.0
    assert per_class[NEUTRAL].precision == 0.0    # predicted once, wrongly
    assert per_class[NEUTRAL].support == 0


# ---------------------------------------------------------------- AUROC

# 2 positives, 3 negatives, no ties. Concordant pairs by hand:
#   .9 > .8 .4 .3  -> 3
#   .6 < .8, > .4 .3 -> 2
#   5 of 6 -> 0.8333...
SIMPLE_Y = [1, 0, 1, 0, 0]
SIMPLE_S = [0.9, 0.8, 0.6, 0.4, 0.3]


def test_auroc_counts_concordant_pairs():
    assert m.auroc(SIMPLE_Y, SIMPLE_S) == pytest.approx(5 / 6)


def test_auroc_of_a_perfect_and_an_inverted_ranking():
    assert m.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert m.auroc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.0)


def test_ties_count_as_half_a_pair():
    """y = [1,0,1,0], s = [.5,.5,.9,.1]:
        (.5 vs .5) tie = .5, (.5 vs .1) = 1, (.9 vs .5) = 1, (.9 vs .1) = 1
        -> 3.5 / 4 = 0.875
    """
    assert m.auroc([1, 0, 1, 0], [0.5, 0.5, 0.9, 0.1]) == pytest.approx(0.875)


def test_all_scores_tied_is_exactly_chance():
    """A saturated NLI head does this, so it must not come out as 1.0."""
    assert m.auroc([1, 0, 1, 0], [0.99] * 4) == pytest.approx(0.5)


def test_auroc_is_nan_when_a_class_is_missing():
    """Not 0.5 -- chance is a measurement, and this is the absence of one."""
    assert np.isnan(m.auroc([1, 1, 1], [0.9, 0.5, 0.1]))
    assert np.isnan(m.auroc([0, 0, 0], [0.9, 0.5, 0.1]))


def test_auroc_ignores_the_scale_of_the_scores():
    assert m.auroc(SIMPLE_Y, [s * 100 for s in SIMPLE_S]) == pytest.approx(5 / 6)


def test_a_third_label_value_raises():
    with pytest.raises(ValueError, match="binary labels"):
        m.auroc([0, 1, 2], [0.1, 0.2, 0.3])


def test_nan_scores_raise_instead_of_ranking_unpredictably():
    with pytest.raises(ValueError, match="NaN"):
        m.auroc([1, 0], [0.5, float('nan')])


def test_score_count_must_match_label_count():
    with pytest.raises(ValueError, match="expected 2 scores"):
        m.auroc([1, 0], [0.5, 0.4, 0.3])


# ------------------------------------------------------------- ROC curve

def test_roc_curve_points_are_the_hand_drawn_ones():
    """Sorted desc: .9(P) .8(N) .6(P) .4(N) .3(N), with 2 P and 3 N.
        (0,0) -> (0,.5) -> (1/3,.5) -> (1/3,1) -> (2/3,1) -> (1,1)
    """
    roc = m.roc_curve(SIMPLE_Y, SIMPLE_S)

    assert roc.fpr.tolist() == pytest.approx([0, 0, 1 / 3, 1 / 3, 2 / 3, 1])
    assert roc.tpr.tolist() == pytest.approx([0, 0.5, 0.5, 1, 1, 1])
    assert roc.thresholds[0] == np.inf
    assert roc.thresholds[1:].tolist() == pytest.approx([0.9, 0.8, 0.6, 0.4, 0.3])


def test_tied_scores_collapse_into_one_roc_point():
    """No threshold separates equal scores, so the intermediate point is not
    reachable and must not be drawn: .9(P) .5(P) .5(N) .1(N) gives 3 points
    after the origin, not 4."""
    roc = m.roc_curve([1, 0, 1, 0], [0.5, 0.5, 0.9, 0.1])

    assert len(roc.thresholds) == 4
    assert roc.fpr.tolist() == pytest.approx([0, 0, 0.5, 1.0])
    assert roc.tpr.tolist() == pytest.approx([0, 0.5, 1.0, 1.0])


@pytest.mark.parametrize("y, s", [
    (SIMPLE_Y, SIMPLE_S),
    ([1, 0, 1, 0], [0.5, 0.5, 0.9, 0.1]),                       # ties
    ([1, 1, 0, 0, 1, 0], [0.8, 0.8, 0.8, 0.2, 0.2, 0.1]),       # heavy ties
])
def test_trapezoid_under_the_curve_equals_the_rank_auroc(y, s):
    """The two are the same quantity computed two ways, ties included. If they
    ever disagree, the tie handling in one of them is wrong."""
    roc = m.roc_curve(y, s)

    assert np.trapezoid(roc.tpr, roc.fpr) == pytest.approx(roc.auroc)
    assert roc.auroc == pytest.approx(m.auroc(y, s))


def test_the_threshold_at_each_point_reproduces_its_rates():
    """`score >= thresholds[i]` is the stated operating point; check it holds."""
    roc = m.roc_curve(SIMPLE_Y, SIMPLE_S)
    y = np.array(SIMPLE_Y, dtype=bool)
    s = np.array(SIMPLE_S)

    for fpr, tpr, threshold in zip(roc.fpr, roc.tpr, roc.thresholds):
        predicted = s >= threshold
        assert (predicted & y).sum() / y.sum() == pytest.approx(tpr)
        assert (predicted & ~y).sum() / (~y).sum() == pytest.approx(fpr)


def test_roc_needs_both_classes():
    with pytest.raises(ValueError, match="needs both classes"):
        m.roc_curve([1, 1], [0.9, 0.8])


# ------------------------------------------------- within-response AUROC

# The confound, in miniature. The scorer reads only which response a sentence
# came from: every sentence in r1 scores .9, every sentence in r2 scores .1.
# It knows nothing about any individual sentence.
#   r1: labels 1,1,0  all .9   -> within: all tied -> 0.5
#   r2: labels 1,0,0  all .1   -> within: all tied -> 0.5
# Pooled, over 3 positives x 3 negatives = 9 pairs:
#   .9 pos vs {.9 neg, .1 neg, .1 neg} = .5 + 1 + 1 = 2.5   (twice)
#   .1 pos vs {.9 neg, .1 neg, .1 neg} =  0 + .5 + .5 = 1.0
#   -> 6/9 = 0.6667
CONFOUND_GROUPS = ['r1', 'r1', 'r1', 'r2', 'r2', 'r2']
CONFOUND_Y = [1, 1, 0, 1, 0, 0]
CONFOUND_S = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1]


def test_pooled_beats_within_when_scores_only_read_the_response():
    """Why both numbers are reported. A difficulty reader clears chance pooled
    and sits exactly at chance within-response."""
    report = m.detection_report(CONFOUND_Y, CONFOUND_S, groups=CONFOUND_GROUPS)

    assert report.pooled_auroc == pytest.approx(2 / 3)
    assert report.within.macro == pytest.approx(0.5)
    assert report.within.pair_weighted == pytest.approx(0.5)
    assert report.pooled_minus_within == pytest.approx(2 / 3 - 0.5)


def test_within_group_auroc_scores_each_group_separately():
    """r1 ranks perfectly (1.0), r2 is inverted (0.0)."""
    y = [1, 1, 0, 0, 1, 0]
    s = [0.9, 0.8, 0.4, 0.3, 0.1, 0.7]
    groups = ['r1', 'r1', 'r1', 'r1', 'r2', 'r2']

    within = m.within_group_auroc(y, s, groups)

    per_group = {g.group: g for g in within.per_group}
    assert per_group['r1'].auroc == pytest.approx(1.0)
    assert per_group['r2'].auroc == pytest.approx(0.0)
    # macro weights the 2-sentence response as heavily as the 4-sentence one;
    # pair-weighted gives it its 1 pair against r1's 4 -> 4/5.
    assert within.macro == pytest.approx(0.5)
    assert within.pair_weighted == pytest.approx(0.8)
    assert within.n_pairs == 5


def test_single_class_responses_are_skipped_and_counted():
    """An all-supported response admits no ranking. Dropping it silently would
    turn 'AUROC over 1 of 3 responses' into 'AUROC over 3 responses'."""
    y = [1, 1, 0, 0, 1, 0]
    s = [0.9, 0.8, 0.4, 0.3, 0.2, 0.1]
    groups = ['all_pos', 'all_pos', 'all_neg', 'all_neg', 'mixed', 'mixed']

    within = m.within_group_auroc(y, s, groups)

    assert within.n_groups == 3
    assert within.n_groups_used == 1
    assert within.n_groups_skipped == 2
    assert within.macro == pytest.approx(1.0)
    assert [g.group for g in within.per_group] == ['mixed']
    assert '1/3 responses' in within.skipped_note


def test_within_group_auroc_is_nan_when_no_response_is_mixed():
    """The degenerate case the pooled number hides completely: pooled looks
    perfect, and there is not one sentence-level comparison behind it."""
    report = m.detection_report([1, 1, 0, 0], [0.9, 0.9, 0.1, 0.1],
                                groups=['a', 'a', 'b', 'b'])

    assert report.pooled_auroc == pytest.approx(1.0)
    assert np.isnan(report.within.macro)
    assert np.isnan(report.within.pair_weighted)
    assert report.within.n_groups_used == 0
    assert report.within.per_group == ()


def test_one_group_makes_within_equal_pooled():
    within = m.within_group_auroc(SIMPLE_Y, SIMPLE_S, ['only'] * 5)

    assert within.macro == pytest.approx(m.auroc(SIMPLE_Y, SIMPLE_S))
    assert within.pair_weighted == pytest.approx(m.auroc(SIMPLE_Y, SIMPLE_S))


def test_per_group_order_is_first_seen_not_sorted():
    within = m.within_group_auroc([1, 0, 1, 0], [0.9, 0.1, 0.9, 0.1],
                                  ['z', 'z', 'a', 'a'])

    assert [g.group for g in within.per_group] == ['z', 'a']


def test_group_count_must_match_label_count():
    with pytest.raises(ValueError, match="expected 4 group ids"):
        m.within_group_auroc([1, 0, 1, 0], [0.9, 0.1, 0.9, 0.1], ['a', 'a'])


# ------------------------------------------------------------ calibration

def test_ece_and_mce_over_two_bins():
    """s = [.1,.2,.7,.9], y = [0,0,1,1], 2 uniform bins:
        [0,.5): mean .15, observed 0.0 -> gap .15, n 2
        [.5,1]: mean .8,  observed 1.0 -> gap .20, n 2
        ECE = (2*.15 + 2*.20)/4 = .175, MCE = .20
    """
    cal = m.calibration([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9], n_bins=2)

    assert cal.ece == pytest.approx(0.175)
    assert cal.mce == pytest.approx(0.20)
    assert [b.count for b in cal.bins] == [2, 2]
    assert cal.bins[0].mean_score == pytest.approx(0.15)
    assert cal.bins[0].empirical_rate == pytest.approx(0.0)
    assert cal.bins[1].gap == pytest.approx(0.20)


def test_a_perfectly_calibrated_score_has_zero_ece():
    # Half the 0.5-scored items are positive, and nothing else is uncertain.
    cal = m.calibration([1, 0, 1, 1, 0, 0], [0.5, 0.5, 1.0, 1.0, 0.0, 0.0],
                        n_bins=2)

    assert cal.ece == pytest.approx(0.0)
    assert cal.mce == pytest.approx(0.0)


def test_empty_bins_stay_in_the_curve_and_contribute_nothing():
    """4 bins, two of them empty. ECE = (.1 + .1)/2 = .1, not divided by 4."""
    cal = m.calibration([0, 1], [0.1, 0.9], n_bins=4)

    assert [b.count for b in cal.bins] == [1, 0, 0, 1]
    assert np.isnan(cal.bins[1].mean_score)
    assert np.isnan(cal.bins[1].empirical_rate)
    assert np.isnan(cal.bins[1].gap)
    assert cal.ece == pytest.approx(0.1)


def test_the_top_score_lands_in_the_last_bin_not_outside_it():
    cal = m.calibration([1, 1], [1.0, 1.0], n_bins=5)

    assert cal.n == 2
    assert cal.bins[-1].count == 2
    assert cal.ece == pytest.approx(0.0)


def test_a_score_on_a_bin_edge_goes_to_the_upper_bin():
    cal = m.calibration([1], [0.5], n_bins=2)

    assert [b.count for b in cal.bins] == [0, 1]


def test_uniform_bins_reject_scores_outside_zero_one():
    """Uniform bins assume a probability. Raw logits would silently fall out of
    every bin and report ECE 0."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        m.calibration([1, 0], [3.2, -1.5])


def test_quantile_bins_survive_a_fully_saturated_score():
    """Every quantile edge collapses to the same value; that must be one bin
    holding everything, not a crash or a silently empty curve."""
    cal = m.calibration([1, 0, 1, 1], [0.999] * 4, n_bins=10, strategy='quantile')

    assert sum(b.count for b in cal.bins) == 4
    assert cal.ece == pytest.approx(abs(0.75 - 0.999))


def test_quantile_bins_keep_counts_comparable():
    cal = m.calibration([0, 0, 1, 1], [0.05, 0.06, 0.90, 0.99],
                        n_bins=2, strategy='quantile')

    assert [b.count for b in cal.bins] == [2, 2]
    assert cal.strategy == 'quantile'


def test_unknown_bin_strategy_raises():
    with pytest.raises(ValueError, match="unknown bin strategy"):
        m.calibration([1, 0], [0.9, 0.1], strategy='logarithmic')


# --------------------------------------------------------------- assembly

def test_detection_report_carries_the_counts_and_the_orientation():
    report = m.detection_report(CONFOUND_Y, CONFOUND_S, groups=CONFOUND_GROUPS,
                                positive_label='unsupported')

    assert (report.n, report.n_positive, report.n_negative) == (6, 3, 3)
    assert report.positive_label == 'unsupported'
    assert report.roc.auroc == pytest.approx(report.pooled_auroc)
    assert report.calibration.n == 6


def test_without_groups_the_within_number_is_absent_not_faked():
    report = m.detection_report(SIMPLE_Y, SIMPLE_S)

    assert report.within is None
    assert np.isnan(report.pooled_minus_within)
    assert 'not computed' in m.format_report(detection=report)


def test_format_report_renders_both_halves():
    text = m.format_report(
        verdict=m.verdict_report(GOLD, PRED, labels=LABELS),
        detection=m.detection_report(CONFOUND_Y, CONFOUND_S,
                                     groups=CONFOUND_GROUPS,
                                     positive_label='unsupported'),
        title='fixture',
    )

    assert 'AUROC pooled            0.6667' in text
    assert 'pair-weighted over 4 within-response pairs' in text
    assert 'pooled - within' in text
    assert '1 skipped' not in text          # both responses were mixed
    assert 'balanced accuracy 0.7500' in text
    assert CONTRA in text


def test_format_report_handles_an_undefined_within_number():
    """NaN must render as n/a rather than blowing up the run that produced it."""
    report = m.detection_report([1, 1, 0, 0], [0.9, 0.9, 0.1, 0.1],
                                groups=['a', 'a', 'b', 'b'])

    text = m.format_report(detection=report)

    assert 'AUROC within-response   n/a' in text
    assert '2 skipped, single-class' in text


def test_format_reliability_lists_every_bin():
    text = m.format_reliability(m.calibration([0, 1], [0.1, 0.9], n_bins=4))

    assert len(text.splitlines()) == 1 + 4 + 1     # header, bins, footer
    assert 'ECE 0.1000' in text
