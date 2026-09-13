"""E4: derive the low-support floor on train, report it on test.

    python -m eval.calibrate --train eval/results/e4-train-qa \\
                             --test  eval/results/e3-test-full

Reads saved scores from `python -m eval.run score`; never runs the model.

What gets derived, and what only gets audited (PLAN.md D6, decided (b))
-----------------------------------------------------------------------
E3 showed the useful cutoff for *flagging* a sentence sits near
P(entailment) 0.001, while green needs strong support. So the two are separate
gates. This module **derives** one of them:

    verification.low_support_threshold - a sentence is flagged LOW_SUPPORT iff
    P(entailment) < floor. The floor is the largest value whose sentence-level
    false-positive rate on **train** (supported sentences flagged) stays within
    a target fixed before looking, 5% by default. Test is reported, never used
    to choose.

It **audits** the hand-set green gate (ENTAILMENT at `entailment_threshold`)
by measuring how often a green sentence is supported. It also audits the red
underline the app used to have (CONTRADICTION as the argmax class at a gate):
that audit is the evidence P8 removed red on (PLAN.md D7), so it stays in the
report even though no verdict uses it any more.

Derived on QA only: it is the app's setting and the headline task, and E3
measured Data2txt at chance, which would pull a pooled floor anywhere.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from eval import metrics, run

DEFAULT_TARGET_FPR = 0.05
TABLE_TARGETS = (0.01, 0.02, 0.05, 0.10, 0.20)
GREEN_GATES = (0.85, 0.90, 0.95, 0.99)
RED_GATES = (0.85, 0.90, 0.95, 0.99)


# ------------------------------------------------------------------ loading

def scoring_mismatch(meta_a, meta_b):
    """The settings that change P(entailment) itself, where two runs differ.

    Thresholds are deliberately absent: they act on probabilities after
    scoring, and E4 itself adds one, so comparing them would refuse every run
    scored before that commit.
    """
    def key(meta):
        c = meta['config']
        return {'nli_model': c['nli_model'],
                'chunk_size': c['ingestion']['chunk_size'],
                'chunk_overlap': c['ingestion']['chunk_overlap'],
                'max_length': c['verification']['max_length'],
                'segmenter': meta['segmenter']}
    a, b = key(meta_a), key(meta_b)
    return ', '.join(f"{k}: {a[k]!r} vs {b[k]!r}" for k in a if a[k] != b[k])


def load_run(results_dir, task='QA', aggregation='max_entailment'):
    results_dir = Path(results_dir)
    meta = json.loads((results_dir / 'run.json').read_text(encoding='utf-8'))
    if aggregation not in meta['aggregations']:
        raise ValueError(f"{results_dir} has no {aggregation!r} scores; "
                         f"it ran {meta['aggregations']}")
    records = [r for r in run._read_records(results_dir / f'{aggregation}.jsonl.gz')
               if r['task_type'] == task]
    if not records:
        raise ValueError(f"{results_dir} has no {task} responses")
    return meta, records


# -------------------------------------------------------------------- floor

def floor_at_fpr(unsupported, p_entailment, target_fpr):
    """Largest floor `c` with FPR(flag iff P(entailment) < c) <= target.

    Closed form rather than a sweep: with the supported sentences' scores
    sorted ascending and k = floor(target * n), the answer is the (k+1)-th
    smallest. Any larger floor would also flag that sentence and exceed k
    false positives; ties at that value are left unflagged, which can only
    lower the FPR. Returns `inf` when the target allows flagging everything.
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}")
    unsupported = np.asarray(unsupported, dtype=bool)
    p = np.asarray(p_entailment, dtype=float)
    supported = np.sort(p[~unsupported])
    if not len(supported):
        raise ValueError("no supported sentences: FPR is undefined")
    k = int(math.floor(target_fpr * len(supported) + 1e-9))
    return float(supported[k]) if k < len(supported) else float('inf')


def round_floor_down(value, digits=3):
    """Round to `digits` significant figures, never up.

    Rounding up could push the FPR past the target it was derived for; rounding
    down can only lower it. The config value is this rounded number, and every
    reported rate is measured at it rather than at the unrounded one.
    """
    if not math.isfinite(value) or value <= 0:
        return value
    scale = 10 ** (digits - 1 - int(math.floor(math.log10(value))))
    return math.floor(value * scale) / scale


def operating_point(unsupported, p_entailment, floor):
    unsupported = np.asarray(unsupported, dtype=bool)
    flagged = np.asarray(p_entailment, dtype=float) < floor
    tp = int((flagged & unsupported).sum())
    fp = int((flagged & ~unsupported).sum())
    positives, negatives = int(unsupported.sum()), int((~unsupported).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    return {
        'floor': floor,
        'flagged': int(flagged.sum()),
        'fpr': fp / negatives if negatives else 0.0,
        'tpr': recall,
        'precision': precision,
        'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        'base_rate': positives / len(unsupported) if len(unsupported) else 0.0,
    }


def answer_point(records, floor):
    """E3's answer rule with the floor as tau, with its references beside it."""
    gold, score, predicted, empty = run.answer_level(records, floor)
    tp = int((predicted & gold).sum())
    fp = int((predicted & ~gold).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / gold.sum() if gold.sum() else 0.0
    oracle, cutoff = run.oracle_f1(gold, score)
    return {
        'flagged': int(predicted.sum()), 'n': len(gold), 'hallucinated': int(gold.sum()),
        'precision': precision, 'recall': recall,
        'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        'fpr': fp / (~gold).sum() if (~gold).sum() else 0.0,
        'flag_all_f1': run.flag_all_f1(gold),
        'oracle_f1': oracle, 'oracle_cutoff': cutoff,
        'auroc': metrics.auroc(gold, score) if 0 < gold.sum() < len(gold) else float('nan'),
        'empty': empty,
    }


# --------------------------------------------------------------- gate audit

def is_green(sentence, gate):
    """Green under the auditor: P(entailment) at or above the gate."""
    return sentence['p_entailment'] >= gate


def was_red(sentence, gate):
    """Red under the pre-P8 auditor: CONTRADICTION the argmax class, at the gate.

    No verdict uses this any more. It reproduces the removed red underline from
    saved probabilities so the measurement behind D7 can be re-run.
    """
    p_con = sentence['p_contradiction']
    return (p_con >= sentence['p_entailment'] and p_con >= sentence['p_neutral']
            and p_con >= gate)


GATE_KINDS = {
    # name: (predicate, the gold state that makes it right)
    'green': (is_green, False),
    'red': (was_red, True),
}


def gate_audit(sentences, kind, gates):
    """How often a green (or the old red) underline is right, at each gate.

    Green is right when the sentence is supported; red was right when it is
    not. `share` is the fraction of that gold class the gate reaches. Each
    predicate depends only on its own gate, so no other threshold is involved.
    """
    shown, right_when_unsupported = GATE_KINDS[kind]
    relevant = sum(s['unsupported'] == right_when_unsupported for s in sentences)
    rows = []
    for gate in gates:
        chosen = [s for s in sentences if shown(s, gate)]
        correct = sum(s['unsupported'] == right_when_unsupported for s in chosen)
        rows.append({'gate': gate, 'n': len(chosen),
                     'precision': correct / len(chosen) if chosen else float('nan'),
                     'share': correct / relevant if relevant else float('nan')})
    return rows


# -------------------------------------------------------------- reliability

def reliability_svg(cal, title):
    """A reliability diagram as standalone SVG. No plotting dependency.

    Dots are bins (area ~ count) at (mean predicted, observed rate); the dashed
    diagonal is perfect calibration. Drawn on white with fixed colours because
    it is meant to be embedded in a README, not themed.
    """
    size, pad = 360, 48
    plot = size - 2 * pad

    def x(v):
        return pad + v * plot

    def y(v):
        return size - pad - v * plot

    biggest = max((b.count for b in cal.bins), default=1) or 1
    title = (title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size + 20}" '
        f'width="{size}" height="{size + 20}" font-family="sans-serif" font-size="11">',
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{size / 2}" y="20" text-anchor="middle" font-size="13" fill="#222">{title}</text>',
        f'<rect x="{pad}" y="{pad}" width="{plot}" height="{plot}" fill="none" stroke="#999"/>',
        f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="#999" stroke-dasharray="4 4"/>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<text x="{x(tick)}" y="{size - pad + 16}" text-anchor="middle" fill="#444">{tick:g}</text>')
        parts.append(f'<text x="{pad - 8}" y="{y(tick) + 4}" text-anchor="end" fill="#444">{tick:g}</text>')
    points = [(b.mean_score, b.empirical_rate, b.count) for b in cal.bins if b.count]
    if points:
        path = ' '.join(f"{x(m):.1f},{y(r):.1f}" for m, r, _ in points)
        parts.append(f'<polyline points="{path}" fill="none" stroke="#2f6fb3" stroke-width="1.5"/>')
    for mean, rate, count in points:
        radius = 2.5 + 9 * math.sqrt(count / biggest)
        parts.append(f'<circle cx="{x(mean):.1f}" cy="{y(rate):.1f}" r="{radius:.1f}" '
                     f'fill="#2f6fb3" fill-opacity="0.55" stroke="#1d4f86"/>')
    parts += [
        f'<text x="{size / 2}" y="{size - 10}" text-anchor="middle" fill="#222">'
        f'mean predicted P(unsupported) = 1 - P(entailment)</text>',
        f'<text x="14" y="{size / 2}" text-anchor="middle" fill="#222" '
        f'transform="rotate(-90 14 {size / 2})">observed unsupported rate</text>',
        f'<text x="{size - pad}" y="{pad + 16}" text-anchor="end" fill="#222">'
        f'ECE {cal.ece:.3f} (n={cal.n})</text>',
        '</svg>',
    ]
    return '\n'.join(parts) + '\n'


# ------------------------------------------------------------------- report

def _pct(v):
    return 'n/a' if v != v else f"{100 * v:.1f}%"


def build(train_dir, test_dir, target_fpr=DEFAULT_TARGET_FPR, task='QA'):
    train_meta, train = load_run(train_dir, task)
    test_meta, test = load_run(test_dir, task)
    if train_meta['split'] != 'train' or test_meta['split'] != 'test':
        raise ValueError(f"expected a train run and a test run, got "
                         f"{train_meta['split']!r} and {test_meta['split']!r}")
    mismatch = scoring_mismatch(train_meta, test_meta)
    if mismatch:
        raise ValueError(f"train and test runs were scored differently ({mismatch}); "
                         f"a floor derived on one does not transfer to the other")
    # Gates are applied after scoring, so they may differ between runs; the
    # audit re-derives them from saved probabilities.
    green_gate = train_meta['config']['verification']['entailment_threshold']

    tr_gold, tr_p, _, tr_sent = run.sentence_level(train)
    te_gold, te_p, _, te_sent = run.sentence_level(test)

    raw = floor_at_fpr(tr_gold, tr_p, target_fpr)
    floor = round_floor_down(raw)
    result = {
        'task': task, 'target_fpr': target_fpr, 'floor_unrounded': raw, 'floor': floor,
        'train': operating_point(tr_gold, tr_p, floor),
        'test': operating_point(te_gold, te_p, floor),
        'test_answer': answer_point(test, floor),
        'train_answer': answer_point(train, floor),
        'table': [],
        'green': {'train': gate_audit(tr_sent, 'green', GREEN_GATES),
                  'test': gate_audit(te_sent, 'green', GREEN_GATES)},
        'red': {'train': gate_audit(tr_sent, 'red', RED_GATES),
                'test': gate_audit(te_sent, 'red', RED_GATES)},
        'calibration': {
            'train': metrics.calibration(tr_gold, 1.0 - tr_p),
            'test': metrics.calibration(te_gold, 1.0 - te_p),
            'test_quantile': metrics.calibration(te_gold, 1.0 - te_p, strategy='quantile'),
        },
        'provenance': {
            'train': {'run': Path(train_dir).name, 'labels_sha256': train_meta['labels_sha256'],
                      'git': train_meta['git'], 'n_responses': len(train)},
            'test': {'run': Path(test_dir).name, 'labels_sha256': test_meta['labels_sha256'],
                     'git': test_meta['git'], 'n_responses': len(test)},
            'model': train_meta['config']['nli_model'],
            'entailment_threshold': green_gate,
        },
    }
    for target in TABLE_TARGETS:
        f = round_floor_down(floor_at_fpr(tr_gold, tr_p, target))
        result['table'].append({
            'target': target, 'floor': f,
            'train': operating_point(tr_gold, tr_p, f),
            'test': operating_point(te_gold, te_p, f),
            'test_answer': answer_point(test, f),
        })
    return result


def format_result(r):
    p = r['provenance']
    tr, te, ta = r['train'], r['test'], r['test_answer']
    lines = [
        'E4 - low-support floor, derived on train, reported on test',
        '=' * 66,
        f"task {r['task']}  |  max_entailment  |  model {p['model']}",
        f"train run {p['train']['run']} ({p['train']['n_responses']} responses, labels "
        f"{p['train']['labels_sha256'][:12]}..., code {p['train']['git']['commit'][:7]}"
        f"{' dirty' if p['train']['git']['dirty'] else ''})",
        f"test  run {p['test']['run']} ({p['test']['n_responses']} responses, labels "
        f"{p['test']['labels_sha256'][:12]}..., code {p['test']['git']['commit'][:7]}"
        f"{' dirty' if p['test']['git']['dirty'] else ''})",
        f"rule: sentence LOW_SUPPORT iff P(entailment) < floor; floor = largest value with "
        f"train sentence FPR <= {r['target_fpr']:.0%} (target fixed before looking), "
        f"rounded down to 3 significant figures",
        '',
        f"FLOOR  verification.low_support_threshold = {r['floor']}  "
        f"(unrounded {r['floor_unrounded']:.6g})",
        f"  train  FPR {_pct(tr['fpr'])}  TPR {_pct(tr['tpr'])}  precision {_pct(tr['precision'])}"
        f"  flagged {tr['flagged']}  (base rate {_pct(tr['base_rate'])})",
        f"  test   FPR {_pct(te['fpr'])}  TPR {_pct(te['tpr'])}  precision {_pct(te['precision'])}"
        f"  flagged {te['flagged']}  (base rate {_pct(te['base_rate'])})",
        f"  verify: test FPR {_pct(te['fpr'])} against the {r['target_fpr']:.0%} target "
        f"({100 * (te['fpr'] - r['target_fpr']):+.1f} points)",
        '',
        'Answer level on test (answer flagged iff any sentence is below the floor)',
        f"  F1 {100 * ta['f1']:.1f}  (P {100 * ta['precision']:.1f}, R {100 * ta['recall']:.1f}, "
        f"answer FPR {_pct(ta['fpr'])})  flag-all F1 {100 * ta['flag_all_f1']:.1f}  "
        f"test-oracle ceiling {100 * ta['oracle_f1']:.1f}  AUROC {ta['auroc']:.3f}",
        f"  flagged {ta['flagged']} of {ta['n']} answers; {ta['hallucinated']} are hallucinated",
        '',
        'Operating table - floor derived on train at each target, measured on test',
        f"{'target':>7}{'floor':>10}{'train FPR':>11}{'test FPR':>10}{'test TPR':>10}"
        f"{'test prec':>11}{'ans F1':>8}{'ans P':>7}{'ans R':>7}{'flag-all':>10}",
    ]
    for row in r['table']:
        a = row['test_answer']
        lines.append(
            f"{row['target']:>7.0%}{row['floor']:>10.4g}{_pct(row['train']['fpr']):>11}"
            f"{_pct(row['test']['fpr']):>10}{_pct(row['test']['tpr']):>10}"
            f"{_pct(row['test']['precision']):>11}{100 * a['f1']:>8.1f}"
            f"{100 * a['precision']:>7.1f}{100 * a['recall']:>7.1f}{100 * a['flag_all_f1']:>10.1f}")

    for name, what, meaning, status in (
            ('green', 'ENTAILMENT, P(entailment) >= gate', 'supported',
             f"Current gate {p['entailment_threshold']}."),
            ('red', 'pre-P8 CONTRADICTION, argmax at gate', 'unsupported',
             'Removed in P8 (PLAN.md D7); kept as the evidence for that.')):
        lines += ['', f"Gate audit - {name} ({what}); precision = share of {name} "
                      f"sentences that are actually {meaning}; reach = share of {meaning} "
                      f"sentences shown {name}. {status}",
                  f"{'gate':>6}{'train n':>9}{'train prec':>12}{'train reach':>13}"
                  f"{'test n':>8}{'test prec':>11}{'test reach':>12}"]
        for a, b in zip(r[name]['train'], r[name]['test']):
            lines.append(f"{a['gate']:>6}{a['n']:>9}{_pct(a['precision']):>12}{_pct(a['share']):>13}"
                         f"{b['n']:>8}{_pct(b['precision']):>11}{_pct(b['share']):>12}")

    c = r['calibration']
    lines += ['', 'Calibration of 1 - P(entailment) as P(unsupported)',
              f"  ECE train {c['train'].ece:.3f}  test {c['test'].ece:.3f}  "
              f"(15 uniform bins; test with quantile bins {c['test_quantile'].ece:.3f})",
              '', 'Reliability, test (uniform bins)', metrics.format_reliability(c['test'])]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m eval.calibrate')
    parser.add_argument('--train', required=True, help='results dir of a train-split run')
    parser.add_argument('--test', required=True, help='results dir of a test-split run')
    parser.add_argument('--target-fpr', type=float, default=DEFAULT_TARGET_FPR)
    parser.add_argument('--task', default='QA', choices=run.TASK_ORDER)
    parser.add_argument('--out', default=str(run.RESULTS_DIR / 'e4-calibration'))
    args = parser.parse_args(argv)

    result = build(args.train, args.test, args.target_fpr, args.task)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = format_result(result)
    (out / 'report.txt').write_text(text, encoding='utf-8')
    for split in ('train', 'test'):
        (out / f'reliability_{split}.svg').write_text(
            reliability_svg(result['calibration'][split],
                            f"RAGTruth {split} {args.task}: reliability of 1 - P(entailment)"),
            encoding='utf-8')
    (out / 'floor.json').write_text(json.dumps({
        'low_support_threshold': result['floor'],
        'unrounded': result['floor_unrounded'],
        'target_fpr': result['target_fpr'],
        'task': result['task'],
        'train': result['train'], 'test': result['test'],
        'provenance': result['provenance'],
    }, indent=2) + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
