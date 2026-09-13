"""Metrics for scoring the verification layer against labelled data.

Pure numpy on purpose: the test suite must stay offline and dependency-light,
and every number computed here is small enough to hand-check against a fixture.

The module has two halves, because the verifier gets scored two different ways.

*Verdict metrics* treat the output as a 3-class label (ENTAILMENT / NEUTRAL /
CONTRADICTION) against a gold label set: confusion matrix, per-class
precision/recall/F1, balanced accuracy. This is what the UI actually renders.

*Detection metrics* throw the verdict away and treat the auditor as a detector:
one continuous score per sentence, swept over every threshold. That is the
honest way to compare against published RAGTruth numbers, and it is the only
way to ask whether `entailment_threshold` sits anywhere near a good operating
point -- which is what E4 needs.

Pooled vs. within-response AUROC
--------------------------------
`detection_report` reports AUROC two ways and they are not interchangeable.

*Pooled* AUROC ranks every sentence in the eval set against every other one.
*Within-response* AUROC ranks sentences only against their siblings in the same
response, then averages.

They come apart whenever responses differ in difficulty. If some responses are
uniformly supported and others uniformly hallucinated, a model that has learned
nothing about individual sentences -- but reads how hard the *response* is --
scores well pooled and at chance within-response. Pooled AUROC alone cannot
tell that model apart from a real verifier, and RAGTruth has many sentences per
response, so the confound is live rather than hypothetical.

Within-response is the number that says whether the verifier discriminates
sentences. Pooled is reported alongside it because the published baselines are
pooled and dropping it would make the comparison dishonest in the other
direction. A large pooled-minus-within gap is a finding, not a footnote.

Responses whose sentences all share one label admit no within-response ranking
at all; they are skipped, and the count of skipped responses is reported, since
"AUROC over the 14 of 20 responses that were mixed" is a different claim from
"AUROC over 20 responses".
"""
from dataclasses import dataclass

import numpy as np

# Reliability-diagram bins. 15 equal-width bins is the convention ECE is
# usually reported under (Guo et al. 2017); stated here so a run that changes
# it has to say so.
DEFAULT_CALIBRATION_BINS = 15

BIN_STRATEGIES = ('uniform', 'quantile')


# ------------------------------------------------------------------ helpers

def _as_indicator(y_true):
    """Coerce labels to a boolean positive/negative array.

    Accepts bools or 0/1 ints, and rejects a third value outright. A
    silently-dropped third class would show up as an innocuous-looking AUROC
    rather than as an error.
    """
    y = np.asarray(y_true)
    if y.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {y.shape}")
    if y.dtype == bool:
        return y
    values = set(np.unique(y).tolist())
    if not values <= {0, 1}:
        raise ValueError(
            f"binary labels must be 0/1 or bool, got values {sorted(values)}. "
            f"Map the gold labels to a single positive class first."
        )
    return y.astype(bool)


def _as_scores(scores, n):
    s = np.asarray(scores, dtype=float)
    if s.shape != (n,):
        raise ValueError(f"expected {n} scores, got shape {s.shape}")
    if not np.isfinite(s).all():
        # NaN sorts unpredictably and would quietly corrupt every rank below.
        raise ValueError("scores contain NaN or inf")
    return s


def _average_ranks(values):
    """1-based ranks, ties sharing their average rank.

    Tie handling is load-bearing rather than pedantic: an NLI head saturates,
    so entailment probabilities of exactly 1.0 to float precision are common,
    and breaking those ties by input order would inflate AUROC.
    """
    order = np.argsort(values, kind='mergesort')
    ordered = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(ordered):
        stop = start
        while stop + 1 < len(ordered) and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start:stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


# --------------------------------------------------------- verdict metrics

@dataclass(frozen=True)
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int          # gold count
    predicted: int        # predicted count


@dataclass(frozen=True)
class ConfusionMatrix:
    labels: tuple
    counts: np.ndarray    # rows = gold, columns = predicted

    def to_text(self):
        names = [str(x) for x in self.labels]
        width = max(max(len(n) for n in names), 6)
        # Columns must clear the widest *header*, not just the widest count, or
        # the header row runs together and stops lining up with the data.
        cell = max(max(len(n) for n in names),
                   len(str(int(self.counts.max(initial=0))))) + 2
        head = ' ' * (width + 2) + ''.join(f"{n:>{cell}}" for n in names)
        lines = [f"{'':>{width}}  predicted ->", head]
        for name, row in zip(names, self.counts):
            lines.append(f"{name:>{width}}  "
                         + ''.join(f"{int(v):>{cell}}" for v in row))
        return '\n'.join(lines)


def confusion_matrix(y_true, y_pred, labels=None):
    """Gold-by-predicted counts.

    `labels` fixes the row/column order; without it the union of observed
    labels is used, sorted. Pass it explicitly for a report, so a class that
    never appears in a run still gets a row instead of silently vanishing.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"length mismatch: {len(y_true)} gold, {len(y_pred)} predicted")
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    labels = tuple(labels)
    unknown = (set(y_true) | set(y_pred)) - set(labels)
    if unknown:
        raise ValueError(f"labels {sorted(unknown)} are absent from {list(labels)}")

    position = {label: i for i, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)), dtype=int)
    for gold, pred in zip(y_true, y_pred):
        counts[position[gold], position[pred]] += 1
    return ConfusionMatrix(labels=labels, counts=counts)


def per_class_metrics(cm):
    """Precision/recall/F1 per class.

    A class the model never predicts has undefined precision; a class with no
    gold instances has undefined recall. Both are reported as 0.0, which is the
    convention -- read them next to `predicted` and `support`, which say which
    denominator was empty.
    """
    out = []
    for i, label in enumerate(cm.labels):
        tp = int(cm.counts[i, i])
        support = int(cm.counts[i].sum())
        predicted = int(cm.counts[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        out.append(ClassMetrics(label, precision, recall, f1, support, predicted))
    return tuple(out)


def accuracy(cm):
    total = int(cm.counts.sum())
    return float(np.trace(cm.counts)) / total if total else 0.0


def balanced_accuracy(cm):
    """Mean per-class recall, over the classes that actually occur in gold.

    Classes with no gold instances are excluded rather than counted as a recall
    of 0 -- otherwise adding an unused label to `labels` would change the score.
    """
    recalls = [cm.counts[i, i] / cm.counts[i].sum()
               for i in range(len(cm.labels)) if cm.counts[i].sum()]
    return float(np.mean(recalls)) if recalls else 0.0


def macro_f1(per_class):
    """Unweighted mean F1, over classes with gold support.

    Unweighted because the interesting classes here are the rare ones: a corpus
    that is mostly ENTAILMENT would let a micro average hide every
    CONTRADICTION the verifier misses.
    """
    scored = [c.f1 for c in per_class if c.support]
    return float(np.mean(scored)) if scored else 0.0


@dataclass(frozen=True)
class VerdictReport:
    confusion: ConfusionMatrix
    per_class: tuple
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    n: int


def verdict_report(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = per_class_metrics(cm)
    return VerdictReport(
        confusion=cm,
        per_class=per_class,
        accuracy=accuracy(cm),
        balanced_accuracy=balanced_accuracy(cm),
        macro_f1=macro_f1(per_class),
        n=int(cm.counts.sum()),
    )


# ------------------------------------------------------- detection metrics

@dataclass(frozen=True)
class ROCCurve:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray   # score >= threshold predicts positive
    auroc: float


def auroc(y_true, scores):
    """Area under the ROC, computed from ranks (Mann-Whitney U).

    Equivalent to the trapezoidal area under `roc_curve`, including when scores
    tie, and it needs no curve. Returns NaN when either class is absent: AUROC
    is undefined there, and any substitute value (0.5 especially) would be
    indistinguishable from a real measurement downstream.
    """
    y = _as_indicator(y_true)
    s = _as_scores(scores, len(y))
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        return float('nan')
    ranks = _average_ranks(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve(y_true, scores):
    """ROC points, one per distinct score, plus the (0, 0) origin.

    `thresholds[i]` is the lowest score still predicted positive at that point,
    so the operating point is `score >= thresholds[i]`. The origin gets `inf`:
    predict nothing positive.
    """
    y = _as_indicator(y_true)
    s = _as_scores(scores, len(y))
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        raise ValueError(
            f"an ROC needs both classes; got {n_pos} positive and {n_neg} negative"
        )

    order = np.argsort(-s, kind='mergesort')
    s_sorted = s[order]
    y_sorted = y[order]
    # Keep only the last index of each run of equal scores: no threshold can
    # separate tied sentences, so intermediate points are not reachable.
    last_of_run = np.r_[np.diff(s_sorted) != 0, True]

    tps = np.cumsum(y_sorted)[last_of_run]
    fps = np.cumsum(~y_sorted)[last_of_run]
    return ROCCurve(
        fpr=np.r_[0.0, fps / n_neg],
        tpr=np.r_[0.0, tps / n_pos],
        thresholds=np.r_[np.inf, s_sorted[last_of_run]],
        auroc=auroc(y, s),
    )


@dataclass(frozen=True)
class GroupAUROC:
    group: object
    auroc: float
    n_positive: int
    n_negative: int


@dataclass(frozen=True)
class WithinGroupAUROC:
    macro: float              # unweighted mean over usable groups
    pair_weighted: float      # concordance over within-group pairs only
    n_groups: int
    n_groups_used: int
    n_groups_skipped: int     # single-class: no ranking to do
    n_pairs: int              # comparable pairs inside groups
    per_group: tuple

    @property
    def skipped_note(self):
        return (f"macro over {self.n_groups_used}/{self.n_groups} responses; "
                f"{self.n_groups_skipped} skipped, single-class")


def within_group_auroc(y_true, scores, groups):
    """AUROC computed inside each group, then averaged two ways.

    `macro` weights every response equally. `pair_weighted` weights a response
    by how many positive-negative pairs it contributes, which makes it the
    direct analogue of the pooled number: both are a fraction of concordant
    pairs, and they differ only in which pairs are allowed to be compared.
    Report both -- macro is dominated by short responses, pair-weighted by long
    ones, and neither is the obviously right summary.
    """
    y = _as_indicator(y_true)
    s = _as_scores(scores, len(y))
    groups = list(groups)
    if len(groups) != len(y):
        raise ValueError(f"expected {len(y)} group ids, got {len(groups)}")

    # dict preserves first-seen order, so the per-group table is stable.
    seen = {}
    for i, group in enumerate(groups):
        seen.setdefault(group, []).append(i)

    per_group = []
    skipped = 0
    for group, members in seen.items():
        idx = np.asarray(members)
        n_pos = int(y[idx].sum())
        n_neg = len(idx) - n_pos
        if not n_pos or not n_neg:
            skipped += 1
            continue
        per_group.append(GroupAUROC(group, auroc(y[idx], s[idx]), n_pos, n_neg))

    if not per_group:
        return WithinGroupAUROC(float('nan'), float('nan'), len(seen), 0,
                                skipped, 0, ())

    weights = np.array([g.n_positive * g.n_negative for g in per_group], dtype=float)
    values = np.array([g.auroc for g in per_group], dtype=float)
    return WithinGroupAUROC(
        macro=float(values.mean()),
        pair_weighted=float((values * weights).sum() / weights.sum()),
        n_groups=len(seen),
        n_groups_used=len(per_group),
        n_groups_skipped=skipped,
        n_pairs=int(weights.sum()),
        per_group=tuple(per_group),
    )


# ------------------------------------------------------------- calibration

@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_score: float        # NaN when the bin is empty
    empirical_rate: float    # NaN when the bin is empty

    @property
    def gap(self):
        if not self.count:
            return float('nan')
        return abs(self.empirical_rate - self.mean_score)


@dataclass(frozen=True)
class Calibration:
    bins: tuple
    ece: float
    mce: float
    n: int
    strategy: str


def calibration(y_true, scores, n_bins=DEFAULT_CALIBRATION_BINS, strategy='uniform'):
    """Reliability-curve data, plus expected and maximum calibration error.

    ECE is the count-weighted mean gap between a bin's mean score and the
    fraction of that bin that is actually positive; MCE is the worst bin. This
    is the binary form -- calibration of the *score*, not of a classifier's
    confidence in whichever class it happened to pick -- because the score is
    what E4 thresholds.

    `uniform` bins are the ECE convention and show where the model piles up;
    `quantile` bins keep counts comparable, which matters for a saturated NLI
    head that puts most of its mass in the top bin. Empty bins stay in the
    curve, with NaN, and contribute nothing to ECE.
    """
    if strategy not in BIN_STRATEGIES:
        raise ValueError(
            f"unknown bin strategy {strategy!r}; expected one of {BIN_STRATEGIES}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    y = _as_indicator(y_true)
    s = _as_scores(scores, len(y))

    if strategy == 'uniform':
        if len(s) and (s.min() < 0.0 or s.max() > 1.0):
            raise ValueError(
                "uniform calibration bins assume scores in [0, 1]; got "
                f"[{s.min():.4g}, {s.max():.4g}]. Use strategy='quantile' for "
                "scores that are not probabilities."
            )
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif len(s):
        # Collapsing duplicate edges is what makes this safe on a saturated
        # score: ten quantiles of a mostly-1.0 column are mostly the same edge.
        edges = np.unique(np.quantile(s, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([edges[0], edges[0]])
    else:
        edges = np.array([0.0, 1.0])

    bins = []
    total_gap = 0.0
    worst = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Half-open bins, except the last, so the maximum score has a home.
        if hi == edges[-1]:
            in_bin = (s >= lo) & (s <= hi)
        else:
            in_bin = (s >= lo) & (s < hi)
        count = int(in_bin.sum())
        if count:
            mean_score = float(s[in_bin].mean())
            rate = float(y[in_bin].mean())
            gap = abs(rate - mean_score)
            total_gap += count * gap
            worst = max(worst, gap)
        else:
            mean_score = rate = float('nan')
        bins.append(CalibrationBin(float(lo), float(hi), count, mean_score, rate))

    return Calibration(
        bins=tuple(bins),
        ece=total_gap / len(s) if len(s) else 0.0,
        mce=worst,
        n=len(s),
        strategy=strategy,
    )


@dataclass(frozen=True)
class DetectionReport:
    positive_label: str
    n: int
    n_positive: int
    n_negative: int
    pooled_auroc: float
    roc: ROCCurve
    within: object             # WithinGroupAUROC, or None if no groups given
    calibration: Calibration

    @property
    def pooled_minus_within(self):
        """The confound, as one number.

        Large and positive means the pooled score is carried by differences
        *between* responses rather than by ranking sentences inside one.
        """
        if self.within is None:
            return float('nan')
        return self.pooled_auroc - self.within.pair_weighted


def detection_report(y_true, scores, groups=None, positive_label='positive',
                     n_bins=DEFAULT_CALIBRATION_BINS, strategy='uniform'):
    """The whole detector block: pooled AUROC, within-response AUROC, ROC, ECE.

    `scores` must be oriented so that higher means more positive -- if the
    positive class is "hallucinated", pass 1 - P(entailment), not
    P(entailment). Nothing here can detect the wrong orientation; it just
    reports an AUROC below 0.5.
    """
    y = _as_indicator(y_true)
    s = _as_scores(scores, len(y))
    return DetectionReport(
        positive_label=positive_label,
        n=len(y),
        n_positive=int(y.sum()),
        n_negative=int(len(y) - y.sum()),
        pooled_auroc=auroc(y, s),
        roc=roc_curve(y, s),
        within=within_group_auroc(y, s, groups) if groups is not None else None,
        calibration=calibration(y, s, n_bins=n_bins, strategy=strategy),
    )


# ------------------------------------------------------------- uncertainty

@dataclass(frozen=True)
class BootstrapCI:
    estimate: float       # the statistic on the full sample
    lower: float
    upper: float
    level: float
    n_resamples: int
    n_valid: int          # resamples on which the statistic was defined
    n_clusters: int

    def to_text(self, places=3, percent=False):
        if self.estimate != self.estimate:
            return 'n/a'
        scale, suffix = (100.0, '%') if percent else (1.0, '')
        p = max(places - 2, 1) if percent else places
        return (f"{scale * self.estimate:.{p}f}{suffix} "
                f"[{scale * self.lower:.{p}f}, {scale * self.upper:.{p}f}]")


def cluster_bootstrap_ci(statistic, groups, n_resamples=2000, level=0.95, seed=0):
    """Percentile CI for `statistic`, resampling whole groups with replacement.

    `statistic(indices)` receives an integer index array into the caller's item
    arrays and returns a float, or NaN where it is undefined (a rate over zero
    items, an AUROC with one class). NaN resamples are dropped and counted in
    `n_valid`, so a CI resting on few defined resamples says so.

    Groups rather than items, because sentences from one answer share a
    question, a retrieval and a generation: resampling sentences independently
    would pretend a 7-sentence answer is 7 independent observations and make
    the interval too narrow.
    """
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("groups must be a non-empty 1-D array")
    names, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(names))]

    estimate = float(statistic(np.arange(len(groups))))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        picked = rng.integers(0, len(members), size=len(members))
        value = float(statistic(np.concatenate([members[k] for k in picked])))
        if value == value:
            values.append(value)
    if not values:
        lower = upper = float('nan')
    else:
        alpha = (1.0 - level) / 2.0
        lower, upper = (float(v) for v in np.quantile(values, [alpha, 1.0 - alpha]))
    return BootstrapCI(estimate, lower, upper, level, n_resamples, len(values), len(members))


# --------------------------------------------------------------- rendering

# Wide enough for the longest detection label, so the numbers form a column.
_LABEL_WIDTH = 24


def _fmt(value, places=4):
    # NaN is the only value not equal to itself, and it means "undefined here",
    # not "zero" -- so it renders as n/a rather than as a number.
    return 'n/a' if value != value else f"{value:.{places}f}"


def format_report(verdict=None, detection=None, title=None):
    """Render the metric block E3 prints. Plain text, no dependencies."""
    lines = []
    if title:
        lines += [title, '=' * len(title)]

    if verdict is not None:
        lines += ['', f"Verdicts (n={verdict.n})", '-' * 40,
                  verdict.confusion.to_text(), '']
        width = max(len(c.label) for c in verdict.per_class)
        lines.append(f"{'label':<{width}}  {'prec':>7}{'rec':>8}{'F1':>8}"
                     f"{'gold':>8}{'pred':>8}")
        for c in verdict.per_class:
            lines.append(f"{c.label:<{width}}  {c.precision:>7.3f}{c.recall:>8.3f}"
                         f"{c.f1:>8.3f}{c.support:>8}{c.predicted:>8}")
        lines += ['', f"accuracy {verdict.accuracy:.4f}   "
                      f"balanced accuracy {verdict.balanced_accuracy:.4f}   "
                      f"macro F1 {verdict.macro_f1:.4f}"]

    if detection is not None:
        d = detection
        lines += ['', f"Detection of {d.positive_label} (n={d.n}: "
                      f"{d.n_positive} positive, {d.n_negative} negative)",
                  '-' * 40]
        label = 'AUROC pooled'
        lines.append(f"{label:<{_LABEL_WIDTH}}{_fmt(d.pooled_auroc)}")
        label = 'AUROC within-response'
        if d.within is None:
            lines.append(f"{label:<{_LABEL_WIDTH}}not computed "
                         "(no response ids supplied)")
        else:
            w = d.within
            lines.append(f"{label:<{_LABEL_WIDTH}}{_fmt(w.macro)}  "
                         f"({w.skipped_note})")
            lines.append(f"{label:<{_LABEL_WIDTH}}{_fmt(w.pair_weighted)}  "
                         f"(pair-weighted over {w.n_pairs} within-response pairs)")
            label = 'pooled - within'
            lines.append(f"{label:<{_LABEL_WIDTH}}{_fmt(d.pooled_minus_within)}  "
                         f"(how much of pooled is between-response difficulty)")
        c = d.calibration
        label = f"ECE ({len(c.bins)} {c.strategy} bins)"
        lines.append(f"{label:<{_LABEL_WIDTH}}{_fmt(c.ece)}   MCE {_fmt(c.mce)}")
    return '\n'.join(lines)


def format_reliability(cal):
    """The reliability curve as a table -- E4 turns this into a diagram."""
    lines = [f"{'bin':>14}{'n':>7}{'mean score':>12}{'observed':>10}{'gap':>8}"]
    for b in cal.bins:
        span = f"[{b.lower:.2f}, {b.upper:.2f}]"
        lines.append(f"{span:>14}{b.count:>7}{_fmt(b.mean_score, 3):>12}"
                     f"{_fmt(b.empirical_rate, 3):>10}{_fmt(b.gap, 3):>8}")
    lines.append(f"ECE {cal.ece:.4f}   MCE {cal.mce:.4f}   n {cal.n}")
    return '\n'.join(lines)
