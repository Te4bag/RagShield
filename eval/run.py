"""E3: run the auditor over RAGTruth and report the E1 metric block.

    python -m eval.run score  --split test [--task-types QA ...] [--limit N]
    python -m eval.run report eval/results/<run> [--tau 0.85]

`score` runs inference once per aggregation strategy and saves every sentence's
probabilities under `eval/results/<run>/` (gitignored, like all generated
data); `report` computes metrics from those files, so re-reporting at another
threshold never re-runs the model. `score` prints the report when it finishes.

Rules fixed before any number was seen (PLAN.md, "E3 rules locked in")
----------------------------------------------------------------------
Sentence score: P(entailment) of the winning chunk, unrounded. The detector
score is `1 - P(entailment)`; the positive class is *unsupported*.

Sentence -> answer: an answer is predicted hallucinated iff its minimum
sentence P(entailment) is below `tau`, and its score is `1 - min P(entailment)`.
With `tau > 0.5` that is exactly "the app would leave at least one sentence
without a green underline". Gold is RAGTruth's own answer label: any span.

Context: every blank-line-separated passage of the RAGTruth context is treated
as a document and chunked with the app's own chunker, and all chunks are
premises. There is no retrieval step - this measures the verifier. QA contexts
are three short passages, so they stay three premises; Summary and Data2txt
contexts are single long texts that get cut at `chunk_size`, which the
published baselines do not do. Only QA is directly comparable to them.
"""
import argparse
import gzip
import json
import random
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from eval import metrics
from eval.datasets import ragtruth

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / 'eval' / 'results'
TASK_ORDER = ('QA', 'Summary', 'Data2txt')
HEADLINE_TASK = 'QA'
HEADLINE_AGGREGATION = 'max_entailment'

ANSWER_RULE = ("answer predicted hallucinated iff min sentence P(entailment) < tau; "
               "answer score = 1 - min P(entailment); gold = RAGTruth has any span")
COMPARABILITY_NOTE = (
    "Summary and Data2txt contexts are chunked at ingestion.chunk_size, unlike the "
    "published baselines, which see the whole context; only QA is directly "
    "comparable to the baseline table. Published figures are example-level F1.")

# Published example-level F1 on RAGTruth, quoted for context only (PLAN.md).
BASELINES = (
    ('Fine-tuned Llama-3-8B (RAG-HAT)', 83.9),
    ('LettuceDetect-large', 79.2),
    ('Fine-tuned Llama-2-13B (RAGTruth paper)', 78.7),
    ('Luna (encoder-based)', 65.4),
    ('GPT-4 prompting', 63.4),
)


class AlignmentError(RuntimeError):
    """The auditor scored different sentences from the ones that were labelled."""


# ------------------------------------------------------------------ scoring

def context_premises(context):
    """The RAGTruth context as premise chunks, cut the way the app cuts documents."""
    from index.chunker import chunk_documents

    passages = [p.strip() for p in context.split('\n\n') if p.strip()]
    docs = [{'doc_id': f'passage{i}', 'text': text, 'source_type': 'ragtruth'}
            for i, text in enumerate(passages)]
    return chunk_documents(docs)


def _pair_token_counts(tokenizer, premise_texts, sentences):
    """Token count of every (premise, sentence) pair, sentence-major order."""
    if not premise_texts or not sentences:
        return np.zeros((len(sentences), len(premise_texts)), dtype=int)
    pairs = [(p, s) for s in sentences for p in premise_texts]
    encoded = tokenizer([p for p, _ in pairs], [s for _, s in pairs], truncation=False)
    lengths = np.array([len(ids) for ids in encoded['input_ids']], dtype=int)
    return lengths.reshape(len(sentences), len(premise_texts))


def score_example(example, auditor):
    """One response through the auditor, joined to its labels."""
    premises = context_premises(example.context)
    with warnings.catch_warnings():
        # The auditor warns once per instance about truncation; here every
        # truncated pair is counted instead, so the warning is only noise.
        warnings.simplefilter('ignore', RuntimeWarning)
        results = auditor.audit_response(example.output, premises)

    scored = [r['sentence'] for r in results]
    labelled = [s.text for s in example.sentences]
    if scored != labelled:
        raise AlignmentError(
            f"response {example.id}: auditor scored {len(scored)} sentences, "
            f"labels describe {len(labelled)}; segmentation has drifted from "
            f"the labels. Rebuild them with `python -m eval.datasets ragtruth build`."
        )

    texts = [p['text'] for p in premises]
    if auditor.aggregation == 'concatenate' and texts:
        texts = ['\n\n'.join(texts)]
    tokens = _pair_token_counts(auditor.model.tokenizer, texts, labelled)

    sentences = []
    for i, (result, label) in enumerate(zip(results, example.sentences)):
        probs = result['probabilities']
        sentences.append({
            'unsupported': label.unsupported,
            'label_types': list(label.label_types),
            'implicit_true': label.implicit_true,
            'due_to_null': label.due_to_null,
            'verdict': result['verdict'],
            # No premise means nothing supports the sentence: P(entailment) 0.
            'p_entailment': probs['ENTAILMENT'] if probs else 0.0,
            'p_contradiction': probs['CONTRADICTION'] if probs else 0.0,
            'p_neutral': probs['NEUTRAL'] if probs else 1.0,
            'max_pair_tokens': int(tokens[i].max()) if tokens.size else 0,
            'truncated_pairs': int((tokens[i] > auditor.max_length).sum()) if tokens.size else 0,
        })
    return {
        'id': example.id,
        'task_type': example.task_type,
        'model': example.model,
        'quality': example.quality,
        'n_spans': example.n_spans,
        'n_premises': len(texts),
        'sentences': sentences,
    }


def score_examples(examples, auditor, aggregation, progress=True):
    from verify.nli_checker import AGGREGATIONS

    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {aggregation!r}; expected one of {AGGREGATIONS}")
    # Set on the instance rather than rebuilding the auditor: construction loads
    # a 568 MB checkpoint, and aggregation is only read inside audit_response.
    auditor.aggregation = aggregation
    records = []
    for n, example in enumerate(examples, 1):
        records.append(score_example(example, auditor))
        if progress and n % 100 == 0:
            print(f"  {aggregation}: {n}/{len(examples)}", file=sys.stderr, flush=True)
    return records


def select_examples(split, task_types=None, limit=None, seed=0):
    examples = ragtruth.load_examples(split, task_types=task_types)
    examples.sort(key=lambda e: (e.task_type, int(e.id) if e.id.isdigit() else e.id))
    if limit is not None and limit < len(examples):
        examples = sorted(random.Random(seed).sample(examples, limit),
                          key=lambda e: (e.task_type, e.id))
    return examples


# ------------------------------------------------------------------ storage

def _write_records(records, path):
    with open(path, 'wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
            for rec in records:
                gz.write((json.dumps(rec, separators=(',', ':')) + '\n').encode('utf-8'))


def _read_records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        return [json.loads(line) for line in fh]


def _git_state():
    def git(*args):
        return subprocess.run(['git', *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    return {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain'))}


def _environment(auditor):
    import sentence_transformers
    import torch
    import transformers
    device = getattr(auditor.model, 'device', None)
    return {
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'sentence_transformers': sentence_transformers.__version__,
        'device': str(device),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run_score(split, task_types, aggregations, limit, seed, name):
    from index import cfg
    from verify import NLIAuditor

    examples = select_examples(split, task_types, limit, seed)
    manifest = json.loads((ragtruth.DATA_DIR / 'manifest.json').read_text(encoding='utf-8'))
    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    auditor = NLIAuditor()
    meta = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'split': split,
        'task_types': list(task_types) if task_types else list(TASK_ORDER),
        'limit': limit,
        'seed': seed,
        'n_responses': len(examples),
        'aggregations': list(aggregations),
        'git': _git_state(),
        'dataset': {'repo': ragtruth.REPO_ID, 'revision': ragtruth.REVISION},
        # Quoted with every number: the labels are not in the repo, so this is
        # how a rebuild is checked against the run that produced a result.
        'labels_sha256': manifest['sha256'][split],
        'segmenter': manifest['segmenter'],
        'config': {
            'nli_model': cfg['models']['nli_model'],
            'verification': dict(cfg['verification']),
            'ingestion': dict(cfg['ingestion']),
        },
        'environment': _environment(auditor),
        'seconds': {},
    }
    for aggregation in aggregations:
        start = time.perf_counter()
        records = score_examples(examples, auditor, aggregation)
        meta['seconds'][aggregation] = round(time.perf_counter() - start, 1)
        _write_records(records, out / f'{aggregation}.jsonl.gz')
    (out / 'run.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    return out


# ---------------------------------------------------------------- reporting

def answer_level(records, tau):
    """Apply the fixed sentence -> answer rule. Returns (gold, score, predicted, n_empty)."""
    gold, score, predicted, empty = [], [], [], 0
    for rec in records:
        gold.append(rec['n_spans'] > 0)
        p = [s['p_entailment'] for s in rec['sentences']]
        if not p:
            # Nothing was scored, so nothing was flagged.
            empty += 1
            score.append(0.0)
            predicted.append(False)
            continue
        score.append(1.0 - min(p))
        predicted.append(min(p) < tau)
    return np.array(gold), np.array(score), np.array(predicted), empty


def sentence_level(records):
    """Flattened sentence arrays: (gold, P(entailment), response ids, sentences)."""
    rows = [(rec['id'], s) for rec in records for s in rec['sentences']]
    gold = np.array([s['unsupported'] for _, s in rows], dtype=bool)
    p_ent = np.array([s['p_entailment'] for _, s in rows], dtype=float)
    groups = [rid for rid, _ in rows]
    return gold, p_ent, groups, [s for _, s in rows]


def _binary(gold, predicted, positive, negative):
    return metrics.verdict_report(
        [positive if g else negative for g in gold],
        [positive if p else negative for p in predicted],
        labels=(positive, negative))


def flag_all_f1(gold):
    """F1 of predicting every item positive: precision = base rate, recall = 1.

    Printed beside every F1 because recall near 1 makes F1 track the base rate.
    At tau 0.85 the verifier flags almost everything, so without this reference
    Data2txt's ~78 would read as a match for LettuceDetect's 79.2 when it is
    exactly what flagging every answer scores.
    """
    gold = np.asarray(gold, dtype=bool)
    rate = gold.mean() if len(gold) else 0.0
    return float(2 * rate / (1 + rate)) if rate else 0.0


def oracle_f1(gold, score):
    """Best F1 over every threshold, chosen on the same labels it is scored on.

    A ceiling, never a result: it says how much F1 a well-placed tau could buy,
    which is E4's question, while E4 must pick tau on train. Returns
    `(f1, min_p_entailment_cutoff)`, the cutoff meaning "flag iff min
    P(entailment) <= cutoff".
    """
    gold = np.asarray(gold, dtype=bool)
    if not 0 < gold.sum() < len(gold):
        return float('nan'), float('nan')
    roc = metrics.roc_curve(gold, score)
    positives, negatives = gold.sum(), len(gold) - gold.sum()
    tp = roc.tpr * positives
    fp = roc.fpr * negatives
    with np.errstate(invalid='ignore', divide='ignore'):
        f1 = np.where(tp > 0, 2 * tp / (2 * tp + fp + (positives - tp)), 0.0)
    best = int(np.argmax(f1))
    return float(f1[best]), float(1.0 - roc.thresholds[best])


def _gold_category(sentence):
    if not sentence['unsupported']:
        return 'supported'
    return 'conflict' if any('Conflict' in t for t in sentence['label_types']) else 'baseless'


def _crosstab(sentences):
    verdicts = ('ENTAILMENT', 'NEUTRAL', 'CONTRADICTION')
    corner = 'gold / verdict'
    lines = [f"{corner:<16}" + ''.join(f"{v:>15}" for v in verdicts) + f"{'n':>8}"]
    for category in ('supported', 'baseless', 'conflict'):
        rows = [s for s in sentences if _gold_category(s) == category]
        if not rows:
            continue
        cells = ''.join(
            f"{sum(s['verdict'] == v for s in rows) / len(rows):>15.1%}" for v in verdicts)
        lines.append(f"{category:<16}{cells}{len(rows):>8}")
    return '\n'.join(lines)


def scope_summary(records, tau):
    """Every number for one (aggregation, task scope), as a dict plus text blocks."""
    gold_s, p_ent, groups, sentences = sentence_level(records)
    gold_a, score_a, pred_a, empty = answer_level(records, tau)
    out = {'n_responses': len(records), 'n_sentences': len(sentences),
           'empty_answers': empty}

    out['sentence_detection'] = (
        metrics.detection_report(gold_s, 1.0 - p_ent, groups=groups,
                                 positive_label='unsupported sentences')
        if 0 < gold_s.sum() < len(gold_s) else None)
    out['sentence_verdict'] = _binary(gold_s, p_ent < tau, 'unsupported', 'supported')

    out['answer_detection'] = (
        metrics.detection_report(gold_a, score_a, positive_label='hallucinated answers')
        if 0 < gold_a.sum() < len(gold_a) else None)
    out['answer_verdict'] = _binary(gold_a, pred_a, 'hallucinated', 'clean')
    out['answer_flag_all_f1'] = flag_all_f1(gold_a)
    out['answer_oracle_f1'], out['answer_oracle_cutoff'] = oracle_f1(gold_a, score_a)
    out['sentence_flag_all_f1'] = flag_all_f1(gold_s)

    # Sensitivity: sentences touched by "true but absent from the source" spans.
    keep = np.array([not s['implicit_true'] for s in sentences], dtype=bool)
    out['implicit_true_sentences'] = int((~keep).sum())
    if keep.any() and 0 < gold_s[keep].sum() < keep.sum():
        kept_groups = [g for g, k in zip(groups, keep) if k]
        out['without_implicit_true'] = metrics.detection_report(
            gold_s[keep], 1.0 - p_ent[keep], groups=kept_groups)
    else:
        out['without_implicit_true'] = None

    pairs = sum(rec['n_premises'] * len(rec['sentences']) for rec in records)
    out['pairs'] = pairs
    out['truncated_pairs'] = sum(s['truncated_pairs'] for s in sentences)
    out['sentences_with_truncation'] = sum(s['truncated_pairs'] > 0 for s in sentences)
    out['crosstab'] = _crosstab(sentences)
    return out


def _fmt(x, places=3):
    return 'n/a' if x is None or x != x else f"{x:.{places}f}"


def _headline_numbers(s):
    ad, sd = s['answer_detection'], s['sentence_detection']
    a = s['answer_verdict'].per_class[0]
    return {
        'answer_f1': a.f1, 'answer_precision': a.precision, 'answer_recall': a.recall,
        'answer_auroc': ad.pooled_auroc if ad else float('nan'),
        'sentence_auroc_pooled': sd.pooled_auroc if sd else float('nan'),
        'sentence_auroc_within': sd.within.pair_weighted if sd and sd.within else float('nan'),
        'sentence_f1': s['sentence_verdict'].per_class[0].f1,
        'answer_flag_all_f1': s['answer_flag_all_f1'],
        'answer_oracle_f1': s['answer_oracle_f1'],
        'answer_oracle_cutoff': s['answer_oracle_cutoff'],
    }


def build_report(meta, records_by_aggregation, tau):
    lines = []
    tasks = [t for t in TASK_ORDER if any(
        r['task_type'] == t for recs in records_by_aggregation.values() for r in recs)]
    scopes = tasks + (['ALL'] if len(tasks) > 1 else [])

    summaries = {}
    for aggregation, records in records_by_aggregation.items():
        for scope in scopes:
            subset = records if scope == 'ALL' else [r for r in records if r['task_type'] == scope]
            summaries[aggregation, scope] = scope_summary(subset, tau)

    lines += ['RagShield on RAGTruth', '=' * 21,
              f"split {meta['split']}  |  responses {meta['n_responses']}"
              + (f" (random sample, seed {meta['seed']})" if meta.get('limit') else '')
              + f"  |  model {meta['config']['nli_model']}",
              f"tau = {tau}  (verification.entailment_threshold"
              + (", hand-picked; E4 derives it)" if tau == meta['config']['verification']['entailment_threshold']
                 else f"; overridden, config has {meta['config']['verification']['entailment_threshold']})"),
              f"rule: {ANSWER_RULE}",
              f"labels sha256 {meta['labels_sha256']}  |  dataset {meta['dataset']['repo']}@{meta['dataset']['revision'][:8]}",
              f"code {meta['git']['commit'][:7]}{' (dirty tree)' if meta['git']['dirty'] else ''}"
              f"  |  {meta['environment']['device']} {meta['environment'].get('gpu') or ''}".rstrip(),
              f"note: {COMPARABILITY_NOTE}", '']

    head_key = (HEADLINE_AGGREGATION, HEADLINE_TASK)
    if head_key in summaries:
        h = _headline_numbers(summaries[head_key])
        s = summaries[head_key]
        lines += [f"HEADLINE - {HEADLINE_TASK}, {HEADLINE_AGGREGATION}", '-' * 40,
                  f"answer-level F1 {100 * h['answer_f1']:.1f}  "
                  f"(P {100 * h['answer_precision']:.1f}, R {100 * h['answer_recall']:.1f}) "
                  f"at tau {tau}; answer AUROC {_fmt(h['answer_auroc'])}",
                  f"  flag-every-answer baseline F1 {100 * h['answer_flag_all_f1']:.1f} "
                  f"({100 * (h['answer_f1'] - h['answer_flag_all_f1']):+.1f} for the rule at tau {tau})",
                  f"  oracle F1 {100 * h['answer_oracle_f1']:.1f} at min P(entailment) <= "
                  f"{_fmt(h['answer_oracle_cutoff'])} - threshold picked ON TEST: a ceiling "
                  f"for E4, not a result",
                  f"sentence AUROC pooled {_fmt(h['sentence_auroc_pooled'])}, "
                  f"within-response {_fmt(h['sentence_auroc_within'])} "
                  f"(pair-weighted, {s['sentence_detection'].within.n_groups_used if s['sentence_detection'] else 0} mixed responses)",
                  '', 'Published example-level F1, for context (whole-context, not tuned to this rule):']
        lines += [f"  {name:<42}{f1:>5.1f}" for name, f1 in BASELINES]
        lines.append('')

    lines += ['ABLATION - aggregation x task', '-' * 40,
              f"{'aggregation':<16}{'scope':<10}{'ans F1':>8}{'flag-all':>10}{'oracle':>8}"
              f"{'ans P':>8}{'ans R':>8}"
              f"{'ans AUROC':>11}{'sent AUROC':>12}{'within':>9}{'sent F1':>9}{'trunc':>8}"]
    for aggregation in records_by_aggregation:
        for scope in scopes:
            s = summaries[aggregation, scope]
            h = _headline_numbers(s)
            trunc = s['truncated_pairs'] / s['pairs'] if s['pairs'] else 0.0
            oracle = h['answer_oracle_f1']
            lines.append(
                f"{aggregation:<16}{scope:<10}{100 * h['answer_f1']:>8.1f}"
                f"{100 * h['answer_flag_all_f1']:>10.1f}"
                f"{('n/a' if oracle != oracle else f'{100 * oracle:.1f}'):>8}"
                f"{100 * h['answer_precision']:>8.1f}{100 * h['answer_recall']:>8.1f}"
                f"{_fmt(h['answer_auroc']):>11}{_fmt(h['sentence_auroc_pooled']):>12}"
                f"{_fmt(h['sentence_auroc_within']):>9}{100 * h['sentence_f1']:>9.1f}{trunc:>8.1%}")
    lines += ['  ans F1 = the fixed rule at tau; flag-all = F1 of calling every answer '
              'hallucinated; oracle = best F1 with tau picked on this same split (ceiling, not a result)',
              '  within = sentence AUROC within-response, pair-weighted; '
              'trunc = share of (premise, sentence) pairs over max_length',
              '  ALL pools three tasks with very different base rates and difficulty; '
              'read the per-task rows', '']

    for aggregation in records_by_aggregation:
        for scope in scopes:
            s = summaries[aggregation, scope]
            title = f"{aggregation} / {scope}"
            lines += ['', '#' * 72, f"# {title}", '#' * 72,
                      f"responses {s['n_responses']}, sentences {s['n_sentences']}, "
                      f"answers with no scorable sentence {s['empty_answers']}, "
                      f"pairs {s['pairs']}, truncated pairs {s['truncated_pairs']} "
                      f"({s['sentences_with_truncation']} sentences affected)"]
            if s['sentence_detection']:
                lines.append(metrics.format_report(
                    verdict=s['sentence_verdict'], detection=s['sentence_detection'],
                    title=f"Sentence level (predicted unsupported iff P(entailment) < {tau})"))
            if s['answer_detection']:
                lines.append(metrics.format_report(
                    verdict=s['answer_verdict'], detection=s['answer_detection'],
                    title='Answer level (' + ANSWER_RULE + ')'))
            lines += ['', 'Verdict by gold category (row %)', s['crosstab']]
            w = s['without_implicit_true']
            if w:
                lines.append(
                    f"\nSensitivity: excluding {s['implicit_true_sentences']} sentences touched "
                    f"by implicit_true spans -> sentence AUROC pooled {_fmt(w.pooled_auroc)}, "
                    f"within {_fmt(w.within.pair_weighted)}")
    return '\n'.join(lines) + '\n'


def run_report(results_dir, tau=None):
    results_dir = Path(results_dir)
    meta = json.loads((results_dir / 'run.json').read_text(encoding='utf-8'))
    if tau is None:
        tau = meta['config']['verification']['entailment_threshold']
    records = {agg: _read_records(results_dir / f'{agg}.jsonl.gz')
               for agg in meta['aggregations']}
    text = build_report(meta, records, tau)
    suffix = '' if tau == meta['config']['verification']['entailment_threshold'] else f'_tau{tau}'
    (results_dir / f'report{suffix}.txt').write_text(text, encoding='utf-8')
    return text


# ----------------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m eval.run')
    sub = parser.add_subparsers(dest='command', required=True)

    s = sub.add_parser('score', help='run the auditor and save per-sentence probabilities')
    s.add_argument('--split', choices=ragtruth.SPLITS, default='test')
    s.add_argument('--task-types', nargs='+', choices=ragtruth.TASK_TYPES)
    s.add_argument('--aggregations', nargs='+',
                   default=['max_entailment', 'concatenate'])
    s.add_argument('--limit', type=int, help='random sample of N responses (dev runs)')
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--name', help='results subdirectory (default: timestamp)')

    r = sub.add_parser('report', help='compute metrics from a saved run')
    r.add_argument('results_dir')
    r.add_argument('--tau', type=float)

    args = parser.parse_args(argv)
    if args.command == 'score':
        name = args.name or datetime.now().strftime('%Y%m%d-%H%M%S')
        out = run_score(args.split, args.task_types, args.aggregations,
                        args.limit, args.seed, name)
        print(run_report(out))
        print(f"saved to {out}", file=sys.stderr)
    else:
        print(run_report(args.results_dir, args.tau))


if __name__ == '__main__':
    main()
