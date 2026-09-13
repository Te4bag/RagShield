"""E5: score the in-domain labelled set and report it, separately from RAGTruth.

    python -m eval.indomain score    # runs the auditor; saves eval/results/e5-indomain/
    python -m eval.indomain report   # recomputes the report from the saved scores

The set is built by `python -m eval.datasets indomain` (see that module and
PLAN.md D4). Its labels were frozen, and their hash recorded in PLAN.md, before
this module scored anything; every report prints the hash it was computed on.

What is measured, and against which RAGTruth number it can be read:

- **Verdict precision on real pipeline answers.** Of the sentences the app
  turns green / yellow / orange, how many are unsupported? On RAGTruth test QA
  that was 1.5% / 8.0% / 21.4% (P8). Rates carry 95% bootstrap CIs over
  questions, because 161 sentences from 60 answers are not 161 independent draws.
- **Ranking**: sentence AUROC of `1 - P(entailment)`, pooled and within-answer.
- **Answer rule** from E3/E4: flagged iff any sentence is orange.
- **Perturbations**: supported sentences edited to be unsupported, scored
  against the same chunks. Real hallucinations are rare in a small set, so this
  is where the orange flag's recall gets tested, and pairing each edit with its
  source shows whether the score moved at all.
- **META sentences** ("I don't know") are never scored as supported or
  unsupported; their verdicts are reported on their own (PLAN.md D2).
"""
import argparse
import json
import sys
import time
import warnings
from datetime import datetime

import numpy as np

from eval import metrics
from eval.datasets import indomain

RESULTS_DIR = indomain.REPO_ROOT / 'eval' / 'results' / 'e5-indomain'
VERDICT_ORDER = ('ENTAILMENT', 'NEUTRAL', 'LOW_SUPPORT')
COLOUR = {'ENTAILMENT': 'green', 'NEUTRAL': 'yellow', 'LOW_SUPPORT': 'orange'}
# Unsupported share of each verdict on RAGTruth test QA under the P8 rule.
RAGTRUTH_UNSUPPORTED = {'ENTAILMENT': 0.015, 'NEUTRAL': 0.080, 'LOW_SUPPORT': 0.214}
N_RESAMPLES = 2000


class AlignmentError(RuntimeError):
    """The auditor scored different sentences from the ones that were labelled."""


# ------------------------------------------------------------------ scoring

def score(raw, auditor):
    """One record per labelled answer sentence and per perturbation."""
    labels = {r['sentence_id']: r for r in raw['sentence_labels']}
    retrieval = {r['question_id']: r['retrieval_has_answer'] for r in raw['question_labels']}
    questions = {q['id']: q for q in raw['questions']}
    answers = {a['question_id']: a for a in raw['answers']}
    source_question = {s['id']: a['question_id'] for a in raw['answers'] for s in a['sentences']}

    def record(qid, item_id, text, result, **fields):
        q = questions[qid]
        return {
            'id': item_id, 'question_id': qid, 'doc': q['doc'], 'batch': q.get('batch'),
            'answerable': q['answerable'], 'retrieval_has_answer': retrieval[qid],
            'text': text, 'verdict': result['verdict'], 'p_entailment': result['entailment'],
            'chunk_id': (result.get('evidence') or {}).get('chunk_id'), **fields,
        }

    records = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for a in raw['answers']:
            results = auditor.audit_response(a['answer'], a['retrieved'])
            scored = [r['sentence'] for r in results]
            labelled = [s['text'] for s in a['sentences']]
            if scored != labelled:
                raise AlignmentError(
                    f"{a['question_id']}: auditor scored {len(scored)} sentences, "
                    f"labels describe {len(labelled)}")
            for s, r in zip(a['sentences'], results):
                lab = labels[s['id']]
                records.append(record(
                    a['question_id'], s['id'], s['text'], r, origin='answer', kind=None,
                    source=None, label=lab['label'], borderline=lab.get('borderline', False),
                    multi_chunk=lab.get('multi_chunk', False)))

        for p in raw['perturbations']:
            qid = source_question[p['source']]
            results = auditor.audit_response(p['sentence'], answers[qid]['retrieved'])
            if len(results) != 1:
                raise AlignmentError(f"{p['id']}: segmented into {len(results)} sentences, expected 1")
            records.append(record(
                qid, p['id'], p['sentence'], results[0], origin='perturbed', kind=p['kind'],
                source=p['source'], label='UNSUPPORTED', borderline=False, multi_chunk=False))
    return records


# ---------------------------------------------------------------- reporting

def _pct(x):
    return 'n/a' if x != x else f"{100 * x:.1f}%"


def rate_ci(groups, subset, hit, seed=0):
    """Share of `hit` among `subset` items, with a CI over question clusters."""
    subset = np.asarray(subset, dtype=bool)
    hit = np.asarray(hit, dtype=bool)

    def stat(idx):
        chosen = idx[subset[idx]]
        return hit[chosen].mean() if len(chosen) else float('nan')

    return metrics.cluster_bootstrap_ci(stat, groups, n_resamples=N_RESAMPLES, seed=seed)


def auroc_ci(groups, gold, score_, seed=0):
    gold = np.asarray(gold, dtype=bool)
    score_ = np.asarray(score_, dtype=float)

    def stat(idx):
        y = gold[idx]
        return metrics.auroc(y, score_[idx]) if 0 < y.sum() < len(y) else float('nan')

    return metrics.cluster_bootstrap_ci(stat, groups, n_resamples=N_RESAMPLES, seed=seed)


def _verdict_table(rows):
    labels = [lab for lab in ('SUPPORTED', 'UNSUPPORTED', 'META', 'NO_CLAIM')
              if any(r['label'] == lab for r in rows)]
    lines = [f"{'label / verdict':<16}" + ''.join(f"{COLOUR[v]:>9}" for v in VERDICT_ORDER)
             + f"{'n':>6}"]
    for lab in labels:
        these = [r for r in rows if r['label'] == lab]
        lines.append(f"{lab:<16}" + ''.join(
            f"{sum(r['verdict'] == v for r in these):>9}" for v in VERDICT_ORDER)
            + f"{len(these):>6}")
    return lines


def _precision_block(rows, title):
    """Unsupported share per verdict, with CIs, beside the RAGTruth figure."""
    groups = [r['question_id'] for r in rows]
    unsupported = [r['label'] == 'UNSUPPORTED' for r in rows]
    lines = [title, f"  {'verdict':<8}{'n':>5}  {'unsupported [95% CI]':<26}{'RAGTruth test QA':>18}"]
    for v in VERDICT_ORDER:
        in_v = [r['verdict'] == v for r in rows]
        ci = rate_ci(groups, in_v, unsupported)
        lines.append(f"  {COLOUR[v]:<8}{sum(in_v):>5}  {ci.to_text(percent=True):<26}"
                     f"{_pct(RAGTRUTH_UNSUPPORTED[v]):>18}")
    return lines


def _detection_lines(rows, label='sentences'):
    gold = np.array([r['label'] == 'UNSUPPORTED' for r in rows])
    s = np.array([1.0 - r['p_entailment'] for r in rows])
    groups = [r['question_id'] for r in rows]
    if not 0 < gold.sum() < len(gold):
        return [f"  AUROC n/a ({label}: one class only)"]
    d = metrics.detection_report(gold, s, groups=groups)
    ci = auroc_ci(groups, gold, s)
    w = d.within
    return [f"  AUROC pooled {ci.to_text()}  ({d.n_positive} unsupported / {d.n_negative} supported {label})",
            f"  AUROC within-answer {w.pair_weighted:.3f} pair-weighted, {w.macro:.3f} macro "
            f"({w.n_groups_used} mixed answers, {w.n_pairs} pairs; no CI - too few answers)"
            if w.n_groups_used else "  AUROC within-answer n/a (no mixed answers)"]


def answer_level(records, floor):
    """E3's rule with the P8 floor: flagged iff any sentence is orange.

    Every sentence of the answer counts toward the minimum, META included,
    because the app underlines those too. Gold: any UNSUPPORTED sentence.
    """
    by_q = {}
    for r in records:
        if r['origin'] == 'answer':
            by_q.setdefault(r['question_id'], []).append(r)
    qids = sorted(by_q)
    gold = np.array([any(r['label'] == 'UNSUPPORTED' for r in by_q[q]) for q in qids])
    min_p = np.array([min(r['p_entailment'] for r in by_q[q]) for q in qids])
    return qids, gold, 1.0 - min_p, min_p < floor


def build_report(records, meta):
    floor = meta['config']['verification']['low_support_threshold']
    green = meta['config']['verification']['entailment_threshold']
    natural = [r for r in records if r['origin'] == 'answer']
    scored = [r for r in natural if r['label'] in indomain.SCORED_LABELS]
    perturbed = [r for r in records if r['origin'] == 'perturbed']
    by_id = {r['id']: r for r in records}
    git = meta['git']

    lines = [
        'RagShield on its own demo PDFs (E5)', '=' * 35,
        f"{len(natural)} answer sentences from {len({r['question_id'] for r in natural})} answers, "
        f"{len(perturbed)} perturbations  |  NLI {meta['config']['nli_model']}  |  "
        f"generator {meta['collection']['generator']['model']}",
        f"labels sha256 {meta['labels_sha256']}  (annotator: {', '.join(meta['annotators'])}; "
        f"frozen before scoring, see PLAN.md E5)",
        f"green >= {green}, orange < {floor}  |  code {git['commit'][:7]}"
        f"{' (dirty tree)' if git['dirty'] else ''}  |  {meta['environment']['device']}",
        f"CIs: 95% percentile bootstrap over questions, {N_RESAMPLES} resamples, seed 0",
        'Labels are relative to the retrieved chunks, not the whole PDF. Never merge with RAGTruth.',
        '',
        'NATURAL ANSWERS - sentence level', '-' * 40,
        *_verdict_table(natural), '',
        *_precision_block(scored, 'Unsupported share of each colour (SUPPORTED/UNSUPPORTED sentences only):'),
    ]

    groups = [r['question_id'] for r in scored]
    unsupported = [r['label'] == 'UNSUPPORTED' for r in scored]
    orange = [r['verdict'] == 'LOW_SUPPORT' for r in scored]
    not_green = [r['verdict'] != 'ENTAILMENT' for r in scored]
    supported = [not u for u in unsupported]
    lines += [
        '',
        f"  orange recall (unsupported flagged orange)   {rate_ci(groups, unsupported, orange).to_text(percent=True)}",
        f"  unsupported not shown green                  {rate_ci(groups, unsupported, not_green).to_text(percent=True)}",
        f"  supported shown green (green reach)          "
        f"{rate_ci(groups, supported, [r['verdict'] == 'ENTAILMENT' for r in scored]).to_text(percent=True)}",
        f"  supported shown orange (orange FPR)          {rate_ci(groups, supported, orange).to_text(percent=True)}",
        '', 'Ranking (score = 1 - P(entailment)):', *_detection_lines(scored),
    ]

    clear = [r for r in scored if not r['borderline']]
    lines += ['', f"Sensitivity - excluding {len(scored) - len(clear)} borderline labels:",
              *[f"  {line.strip()}" for line in _precision_block(clear, '')[1:]],
              *_detection_lines(clear)]

    multi = [r for r in scored if r['multi_chunk']]
    if multi:
        lines += ['', f"Supported only by combining chunks ({len(multi)} sentences; per-chunk scoring "
                      f"cannot entail these by design): "
                  + ', '.join(f"{COLOUR[v]} {sum(r['verdict'] == v for r in multi)}" for v in VERDICT_ORDER)]

    lines += ['', 'By document:']
    for doc in sorted({r['doc'] for r in scored}):
        rows = [r for r in scored if r['doc'] == doc]
        g = [r['question_id'] for r in rows]
        in_green = [r['verdict'] == 'ENTAILMENT' for r in rows]
        uns = [r['label'] == 'UNSUPPORTED' for r in rows]
        lines += [f"  {doc}: {len(rows)} sentences, {sum(uns)} unsupported; green "
                  f"{sum(in_green)}, unsupported among green {rate_ci(g, in_green, uns).to_text(percent=True)}",
                  *[f"  {line}" for line in _detection_lines(rows)]]

    answerable = [r for r in scored if r['answerable']]
    lines += ['', 'Retrieval (answerable questions):']
    for has in (True, False):
        rows = [r for r in answerable if r['retrieval_has_answer'] is has]
        if rows:
            n_q = len({r['question_id'] for r in rows})
            lines.append(f"  retrieved chunks {'contain' if has else 'lack'} the full answer: {n_q} answers, "
                         f"{len(rows)} scored sentences, {_pct(np.mean([r['label'] == 'UNSUPPORTED' for r in rows]))} unsupported")

    lines += ['', 'Errors that matter most (listed, not sampled):']
    false_green = [r for r in scored if r['label'] == 'UNSUPPORTED' and r['verdict'] == 'ENTAILMENT']
    lines.append(f"  unsupported but green ({len(false_green)}):")
    lines += [f"    {r['id']} P={r['p_entailment']:.3f}: {r['text'][:110]}" for r in false_green]
    false_orange = [r for r in scored if r['label'] == 'SUPPORTED' and r['verdict'] == 'LOW_SUPPORT']
    lines.append(f"  supported but orange ({len(false_orange)}):")
    lines += [f"    {r['id']} P={r['p_entailment']:.1e}: {r['text'][:110]}" for r in false_orange]

    qids, gold, a_score, flagged = answer_level(records, floor)
    lines += ['', 'NATURAL ANSWERS - answer level', '-' * 40,
              'rule: flagged iff min sentence P(entailment) < low_support_threshold (any orange sentence); '
              'gold = any UNSUPPORTED sentence; META sentences count toward the minimum']
    if 0 < gold.sum() < len(gold):
        rep = metrics.verdict_report(['hallucinated' if g else 'clean' for g in gold],
                                     ['hallucinated' if f else 'clean' for f in flagged],
                                     labels=('hallucinated', 'clean'))
        c = rep.per_class[0]
        rate = gold.mean()
        lines += [f"  {len(qids)} answers, {int(gold.sum())} with an unsupported sentence, {int(flagged.sum())} flagged",
                  f"  F1 {100 * c.f1:.1f} (P {100 * c.precision:.1f}, R {100 * c.recall:.1f});  "
                  f"flag-all F1 {100 * 2 * rate / (1 + rate):.1f};  answer AUROC {metrics.auroc(gold, a_score):.3f}"]

    meta_rows = [r for r in natural if r['label'] == 'META']
    if meta_rows:
        p = np.array([r['p_entailment'] for r in meta_rows])
        lines += ['', f"META sentences (D2) - {len(meta_rows)}, e.g. 'I don't know'", '-' * 40,
                  '  ' + ', '.join(f"{COLOUR[v]} {sum(r['verdict'] == v for r in meta_rows)}" for v in VERDICT_ORDER)
                  + f";  P(entailment) min {p.min():.1e}, median {np.median(p):.1e}, max {p.max():.1e}"]

    if perturbed:
        lines += ['', 'PERTURBATIONS - supported sentence edited to be unsupported, same chunks', '-' * 40]
        for kind in (None,) + indomain.PERTURBATION_KINDS:
            rows = [r for r in perturbed if kind is None or r['kind'] == kind]
            if not rows:
                continue
            src = [by_id[r['source']] for r in rows]
            g = [r['question_id'] for r in rows]
            everything = [True] * len(rows)
            dropped = [r['p_entailment'] < s['p_entailment'] for r, s in zip(rows, src)]
            green_src = [s['verdict'] == 'ENTAILMENT' for s in src]
            still_green = [r['verdict'] == 'ENTAILMENT' for r in rows]
            name = 'all' if kind is None else kind
            lines += [
                f"  {name} (n={len(rows)}): "
                + ', '.join(f"{COLOUR[v]} {sum(r['verdict'] == v for r in rows)}" for v in VERDICT_ORDER),
                f"    flagged orange          {rate_ci(g, everything, [r['verdict'] == 'LOW_SUPPORT' for r in rows]).to_text(percent=True)}",
                f"    not green               {rate_ci(g, everything, [not x for x in still_green]).to_text(percent=True)}",
                f"    P(entailment) fell      {rate_ci(g, everything, dropped).to_text(percent=True)}",
                f"    still green, of the {sum(green_src)} whose source was green: "
                f"{sum(a and b for a, b in zip(green_src, still_green))}",
            ]
        source_green = [r for r in perturbed if by_id[r['source']]['verdict'] == 'ENTAILMENT']
        if source_green:
            pairs = [(by_id[r['source']]['p_entailment'], r['p_entailment']) for r in source_green]
            lines.append(f"  median P(entailment) for green sources {np.median([a for a, _ in pairs]):.3f} "
                         f"-> perturbed {np.median([b for _, b in pairs]):.3f}")
        lines.append('  perturbed but still green:')
        lines += [f"    {r['id']} ({r['kind']}) P={r['p_entailment']:.3f}: {r['text'][:100]}"
                  for r in perturbed if r['verdict'] == 'ENTAILMENT']
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------- run + CLI

def run_score(out_dir=RESULTS_DIR):
    from eval.run import _environment, _git_state
    from index import cfg
    from verify import NLIAuditor

    raw = indomain.load()
    auditor = NLIAuditor()
    start = time.perf_counter()
    records = score(raw, auditor)
    seconds = round(time.perf_counter() - start, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    indomain.write_jsonl(records, out_dir / 'scores.jsonl')
    meta = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'git': _git_state(),
        'labels_sha256': indomain.label_fingerprint(),
        'annotators': sorted({r['annotator'] for r in raw['sentence_labels']}),
        'collection': json.loads((indomain.DATA_DIR / 'collection.json').read_text(encoding='utf-8')),
        'config': {'nli_model': cfg['models']['nli_model'], 'verification': dict(cfg['verification'])},
        'environment': _environment(auditor),
        'seconds': seconds,
    }
    (out_dir / 'run.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    return run_report(out_dir)


def run_report(out_dir=RESULTS_DIR):
    meta = json.loads((out_dir / 'run.json').read_text(encoding='utf-8'))
    current = indomain.label_fingerprint()
    if meta['labels_sha256'] != current:
        raise ValueError(f"scores were computed on labels {meta['labels_sha256'][:12]}..., "
                         f"but the label files now hash to {current[:12]}...; re-score")
    text = build_report(indomain.read_jsonl(out_dir / 'scores.jsonl'), meta)
    (out_dir / 'report.txt').write_text(text, encoding='utf-8')
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m eval.indomain')
    parser.add_argument('command', choices=('score', 'report'))
    args = parser.parse_args(argv)
    text = run_score() if args.command == 'score' else run_report()
    sys.stdout.reconfigure(encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
