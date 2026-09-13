"""E6: per-stage latency of the pipeline the app runs, cold and warm.

    python -m eval.latency run [--device auto|cpu] [--cold-runs 3] [--reps 3]
                               [--generate 20] [--pace 9] [--name NAME]
    python -m eval.latency report eval/results/e6-latency/<name>

Replaces the README's unbacked "1.2s per query". Every measurement runs in its
own child process, one after another, so a cold start really is cold, the CPU
run cannot see the GPU, and no two torch processes are ever alive at once (the
pagefile on C: cannot take it - see CLAUDE.md).

Stages, in the order `app.py` drives them:

- **cold** (fresh process, repeated `--cold-runs` times): imports, opening the
  index (loads the embedding model), loading the PDFs, the no-op index sync the
  first query triggers, generator and auditor construction, then the first
  retrieve / generate / audit. Their sum is time to the first verified answer,
  excluding Streamlit's own server start.
- **index**: building the index from empty (a fresh clone's first query) and a
  no-op re-sync.
- **warm** (one process, models loaded, a few untimed warm-up calls): retrieve
  over the 60 in-domain questions and audit over their 60 saved answers,
  `--reps` times each - no network, so these are repeatable.
- **layers** (`--generate` live questions): what each layer adds to what a
  user waits. Each question is asked of plain Groq (no context) and through
  RAG (retrieve + Groq with the chunks), in random order, and the RAG answer is
  then checked. RAG minus plain is the retrieval layer's cost; the audit is
  what RagShield's checker adds on top. Differences are paired per question.

Groq's free tier allows 8,000 tokens a minute and one query uses ~1,100, so
live calls are paced `--pace` seconds apart and any 429 is retried after a
back-off. Pacing and back-off are excluded from the latencies and reported as
counts, since the app itself does not retry.
"""
import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / 'eval' / 'results' / 'e6-latency'
WARMUP_CALLS = 3
IDLE_SAMPLES = 8
# The answer every cold run audits first, so cold first-audit numbers compare
# like with like: 4 sentences x 3 chunks.
COLD_AUDIT_QUESTION = 'att04'


# --------------------------------------------------------------- statistics

def summarize(seconds):
    """n, p50, p95, max and mean of a list of durations, in milliseconds."""
    v = np.asarray(seconds, dtype=float) * 1000.0
    if len(v) == 0:
        return {'n': 0, 'p50': float('nan'), 'p95': float('nan'),
                'max': float('nan'), 'mean': float('nan')}
    return {'n': int(len(v)), 'p50': float(np.percentile(v, 50)),
            'p95': float(np.percentile(v, 95)), 'max': float(v.max()),
            'mean': float(v.mean())}


def pace_wait(last_start, now, interval):
    """Seconds to sleep so consecutive calls start at least `interval` apart."""
    if last_start is None:
        return 0.0
    return max(0.0, interval - (now - last_start))


def child_env(device):
    """Environment for a measurement child. `cpu` hides every GPU from torch."""
    env = dict(os.environ)
    env.setdefault('HF_HOME', 'D:/hf-cache')
    if device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device != 'auto':
        raise ValueError(f"device must be 'auto' or 'cpu', got {device!r}")
    return env


# ------------------------------------------------------------ child helpers

class Timer:
    def __init__(self):
        self.steps = {}

    def __call__(self, name, fn, *args, **kwargs):
        start = time.perf_counter()
        value = fn(*args, **kwargs)
        self.steps[name] = time.perf_counter() - start
        return value


def _environment(auditor=None):
    import torch
    info = {
        'python': platform.python_version(),
        'os': platform.platform(),
        'cpu': platform.processor(),
        'cpu_count': os.cpu_count(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if auditor is not None:
        info['auditor_device'] = str(auditor.model.device)
    return info


def _timed_call(call, backoff=15, attempts=8):
    """(value, seconds for the successful attempt, retries by cause).

    Rate limits (429) and dropped connections are retried after a back-off and
    counted, never timed: a full E6 run once died when DNS stopped resolving
    api.groq.com mid-run.
    """
    import groq

    retries = {'rate_limited': 0, 'connection_errors': 0}
    for attempt in range(attempts):
        start = time.perf_counter()
        try:
            value = call()
            return value, time.perf_counter() - start, retries
        except (groq.RateLimitError, groq.APIConnectionError) as err:
            key = 'rate_limited' if isinstance(err, groq.RateLimitError) else 'connection_errors'
            retries[key] += 1
            if attempt == attempts - 1:
                raise
            print(f"    {key.replace('_', ' ')}; retrying in {backoff * (attempt + 1)} s",
                  file=sys.stderr, flush=True)
            time.sleep(backoff * (attempt + 1))


# The no-RAG baseline: the app's generator prompt with the context removed, so
# the same model, temperature and output-format instruction answer from the
# question alone. Keep the format line in step with rag/generator.py.
PLAIN_PROMPT = """
        Answer the user's question.
        Answer in plain prose sentences. Do not use Markdown, bullet points, tables or LaTeX.

        QUESTION:
        {question}

        ANSWER:
        """


def plain_answer(generator, question):
    completion = generator.client.chat.completions.create(
        model=generator.model,
        messages=[{"role": "user", "content": PLAIN_PROMPT.format(question=question)}],
        temperature=0.1,
    )
    return completion.choices[0].message.content


def record_usage(generator):
    """Keep the token usage of the generator client's latest completion.

    Generation time depends on prompt and output length, so the report shows
    tokens beside every Groq timing. Wrapping the client keeps
    `RagGenerator.generate_answer` itself untouched.
    """
    completions = generator.client.chat.completions
    original = completions.create
    last = {}

    def create(*args, **kwargs):
        response = original(*args, **kwargs)
        usage = getattr(response, 'usage', None)
        last['prompt_tokens'] = getattr(usage, 'prompt_tokens', None)
        last['completion_tokens'] = getattr(usage, 'completion_tokens', None)
        return response

    completions.create = create
    return last


def _in_domain():
    from eval.datasets import indomain
    raw = indomain.load(check_segmenter=False)
    questions = {q['id']: q for q in raw['questions']}
    return [(questions[a['question_id']], a) for a in raw['answers']]


def child_build(index_dir, out):
    """Index from empty, then a no-op re-sync."""
    timer = Timer()
    shutil.rmtree(index_dir, ignore_errors=True)
    from ingest import DocumentLoader
    from index import RagShieldIndex
    from eval.datasets.indomain import DOCS_DIR

    index = timer('open_empty_index', RagShieldIndex, db_path=str(index_dir))
    docs = timer('load_documents', DocumentLoader(str(DOCS_DIR)).load)
    plan = timer('build_from_empty', index.sync_documents, docs)
    timer('noop_sync', index.sync_documents, docs)
    result = {'steps': timer.steps, 'chunks': index.collection.count(),
              'added': plan['added'], 'environment': _environment()}
    Path(out).write_text(json.dumps(result, indent=2), encoding='utf-8')


def child_cold(index_dir, out, generate):
    """Fresh process to first verified answer, in app order."""
    timer = Timer()
    timer('import_streamlit', __import__, 'streamlit')
    timer('import_ingest_index', lambda: (__import__('ingest'), __import__('index')))
    timer('import_rag', __import__, 'rag')
    timer('import_verify', __import__, 'verify')      # spaCy model loads here

    from ingest import DocumentLoader
    from index import RagShieldIndex
    from rag import RagGenerator, Retriever
    from verify import NLIAuditor
    from eval.datasets.indomain import DOCS_DIR

    index = timer('open_index', RagShieldIndex, db_path=str(index_dir))
    retriever = Retriever(index=index)
    docs = timer('load_documents', DocumentLoader(str(DOCS_DIR)).load)
    timer('noop_sync', index.sync_documents, docs)
    generator = timer('generator_init', RagGenerator)
    auditor = timer('auditor_init', NLIAuditor)

    question, saved = next((q, a) for q, a in _in_domain() if q['id'] == COLD_AUDIT_QUESTION)
    chunks = timer('first_retrieve', retriever.retrieve, question['question'])
    retries = {}
    if generate:
        context = '\n\n'.join(c['text'] for c in chunks)
        _, seconds, retries = _timed_call(
            lambda: generator.generate_answer(question['question'], context))
        timer.steps['first_generate'] = seconds
    timer('first_audit', auditor.audit_response, saved['answer'], saved['retrieved'])
    timer('second_audit', auditor.audit_response, saved['answer'], saved['retrieved'])
    result = {'steps': timer.steps, 'retries': retries,
              'environment': _environment(auditor)}
    Path(out).write_text(json.dumps(result, indent=2), encoding='utf-8')


def child_warm(index_dir, out, reps, generate, pace, seed):
    from index import RagShieldIndex
    from rag import RagGenerator, Retriever
    from verify import NLIAuditor
    from verify.segmenter import split_into_sentence_spans

    index = RagShieldIndex(db_path=str(index_dir))
    retriever = Retriever(index=index)
    generator = RagGenerator()
    auditor = NLIAuditor()
    items = _in_domain()

    for question, saved in items[:WARMUP_CALLS]:
        retriever.retrieve(question['question'])
        auditor.audit_response(saved['answer'], saved['retrieved'])

    samples = {'retrieve': [], 'audit': [], 'layers': []}
    for rep in range(reps):
        for question, saved in items:
            start = time.perf_counter()
            retriever.retrieve(question['question'])
            samples['retrieve'].append({'id': question['id'], 'rep': rep,
                                        'seconds': time.perf_counter() - start})
        for question, saved in items:
            start = time.perf_counter()
            spans = split_into_sentence_spans(saved['answer'])
            segment = time.perf_counter() - start
            start = time.perf_counter()
            auditor.audit_response(saved['answer'], saved['retrieved'])
            audit = time.perf_counter() - start
            samples['audit'].append({
                'id': question['id'], 'rep': rep, 'seconds': audit, 'segment_seconds': segment,
                'sentences': len(spans), 'pairs': len(spans) * len(saved['retrieved']),
                'answer_chars': len(saved['answer'])})

    # The same answer audited back-to-back and after sitting idle for `pace`
    # seconds. On the laptop GPU an idle pause of 3 s or more made it ~6x slower
    # (scratch check before this was added), and real users pause between
    # questions, so the back-to-back numbers above understate what they wait.
    _, cold_answer = next((q, a) for q, a in items if q['id'] == COLD_AUDIT_QUESTION)
    samples['idle'] = []
    for pause in (0.0, pace):
        for _ in range(IDLE_SAMPLES):
            time.sleep(pause)
            start = time.perf_counter()
            auditor.audit_response(cold_answer['answer'], cold_answer['retrieved'])
            samples['idle'].append({'pause': pause, 'seconds': time.perf_counter() - start})

    # What each layer adds, per question: plain Groq (question only), RAG
    # (retrieve + Groq with the chunks), and the checker (audit of that RAG
    # answer, run straight after it as app.py does). Plain and RAG calls go in
    # a seeded random order per question, so drift in Groq's speed over the run
    # does not land on one of them.
    usage = record_usage(generator)
    rng = random.Random(seed)
    live_questions = [q for q, _ in items]
    rng.shuffle(live_questions)
    last_start, paced = None, 0.0
    retries_total = {'rate_limited': 0, 'connection_errors': 0}

    def paced_call(call):
        nonlocal last_start, paced
        wait = pace_wait(last_start, time.perf_counter(), pace)
        time.sleep(wait)
        paced += wait
        last_start = time.perf_counter()
        value, seconds, retries = _timed_call(call)
        for key, count in retries.items():
            retries_total[key] += count
        return value, seconds, dict(usage)

    for question in live_questions[:generate]:
        q = question['question']
        row = {'id': question['id'], 'plain_first': rng.random() < 0.5}
        for step in (('plain', 'rag') if row['plain_first'] else ('rag', 'plain')):
            if step == 'plain':
                answer, row['plain_generate'], tokens = paced_call(lambda: plain_answer(generator, q))
                row['plain_prompt_tokens'] = tokens.get('prompt_tokens')
                row['plain_completion_tokens'] = tokens.get('completion_tokens')
                row['plain_answer_chars'] = len(answer)
            else:
                start = time.perf_counter()
                chunks = retriever.retrieve(q)
                row['retrieve'] = time.perf_counter() - start
                context = '\n\n'.join(c['text'] for c in chunks)
                answer, row['rag_generate'], tokens = paced_call(
                    lambda: generator.generate_answer(q, context))
                row['rag_prompt_tokens'] = tokens.get('prompt_tokens')
                row['rag_completion_tokens'] = tokens.get('completion_tokens')
                start = time.perf_counter()
                results = auditor.audit_response(answer, chunks)
                row['audit'] = time.perf_counter() - start
                row['sentences'] = len(results)
                row['rag_answer_chars'] = len(answer)
        samples['layers'].append(row)
        print(f"  layers {len(samples['layers'])}/{generate} {question['id']}: plain "
              f"{row['plain_generate']:.2f}s, rag {row['retrieve'] + row['rag_generate']:.2f}s, "
              f"check {row['audit']:.2f}s", file=sys.stderr, flush=True)

    result = {'samples': samples, 'paced_seconds': paced, 'retries': retries_total,
              'environment': _environment(auditor)}
    Path(out).write_text(json.dumps(result, indent=2), encoding='utf-8')


# ------------------------------------------------------------------- parent

def _spawn(args, device):
    subprocess.run([sys.executable, '-m', 'eval.latency', *args],
                   cwd=REPO_ROOT, env=child_env(device), check=True)


def run(device, cold_runs, reps, generate, pace, name, seed=0):
    from eval.run import _git_state
    from index import cfg

    out = RESULTS_DIR / name
    # The report globs cold*.json, so files left by an earlier or failed run
    # would be mixed into this one. Start from nothing.
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    index_dir = out / 'chroma'

    print('index build', file=sys.stderr, flush=True)
    _spawn(['_build', '--index', str(index_dir), '--out', str(out / 'build.json')], device)
    for i in range(cold_runs):
        print(f'cold run {i + 1}/{cold_runs}', file=sys.stderr, flush=True)
        start = time.perf_counter()
        _spawn(['_cold', '--index', str(index_dir), '--out', str(out / f'cold{i}.json'),
                '--generate', '1' if generate else '0'], device)
        cold = json.loads((out / f'cold{i}.json').read_text(encoding='utf-8'))
        cold['process_wall_seconds'] = time.perf_counter() - start
        (out / f'cold{i}.json').write_text(json.dumps(cold, indent=2), encoding='utf-8')
    print('warm', file=sys.stderr, flush=True)
    _spawn(['_warm', '--index', str(index_dir), '--out', str(out / 'warm.json'),
            '--reps', str(reps), '--generate', str(generate), '--pace', str(pace),
            '--seed', str(seed)], device)

    meta = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'device_requested': device, 'cold_runs': cold_runs, 'reps': reps,
        'generate': generate, 'pace_seconds': pace, 'seed': seed,
        'git': _git_state(),
        'config': {'models': dict(cfg['models']), 'retrieval': dict(cfg['retrieval']),
                   'ingestion': dict(cfg['ingestion']),
                   'verification': dict(cfg['verification'])},
    }
    (out / 'run.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    return report(out)


# ------------------------------------------------------------------ report

def _ms(x):
    if x != x:
        return 'n/a'
    return f"{x:,.0f} ms" if abs(x) >= 100 else f"{x:.1f} ms"


def _row(label, stats):
    return (f"  {label:<38}{stats['n']:>5}{_ms(stats['p50']):>12}"
            f"{_ms(stats['p95']):>12}{_ms(stats['max']):>12}")


COLD_STEPS = (
    ('import_streamlit', 'import streamlit'),
    ('import_ingest_index', 'import ingest + index (chromadb, ST)'),
    ('import_rag', 'import rag (groq)'),
    ('import_verify', 'import verify (spaCy model)'),
    ('open_index', 'open index (embedding model)'),
    ('load_documents', 'load the 2 demo PDFs'),
    ('noop_sync', 'index sync, nothing changed'),
    ('generator_init', 'generator init'),
    ('auditor_init', 'auditor init (568 MB checkpoint)'),
    ('first_retrieve', 'first retrieve'),
    ('first_generate', 'first generate (Groq)'),
    ('first_audit', 'first audit (4 sentences)'),
)


def build_report(meta, build, colds, warm):
    env = warm['environment']
    cfg = meta['config']
    git = meta['git']
    lines = [
        'RagShield latency (E6)', '=' * 22,
        f"device requested {meta['device_requested']}; auditor on {env.get('auditor_device')}"
        f"{' (' + env['gpu'] + ')' if env.get('gpu') else ''}; embeddings on cpu (chromadb default)",
        f"torch {env['torch']}  |  python {env['python']}  |  {env['os']}  |  {env['cpu']} x{env['cpu_count']}",
        f"generator {cfg['models']['generator']} via Groq over the network  |  NLI {cfg['models']['nli_model']}  |  "
        f"top_k {cfg['retrieval']['top_k']}, batch_size {cfg['verification']['batch_size']}",
        f"code {git['commit'][:7]}{' (dirty tree)' if git['dirty'] else ''}  |  {meta['created']}",
        'p95 over small n is close to the max; read n with it.', '',
    ]

    header = f"  {'':<38}{'n':>5}{'p50':>12}{'p95':>12}{'max':>12}"
    lines += [f"COLD START - fresh process, {len(colds)} runs", '-' * 40, header]
    total = []
    for key, label in COLD_STEPS:
        values = [c['steps'][key] for c in colds if key in c['steps']]
        if values:
            lines.append(_row(label, summarize(values)))
    for c in colds:
        total.append(sum(v for k, v in c['steps'].items() if k != 'second_audit'))
    lines.append(_row('time to first verified answer', summarize(total)))
    checker_cold = [c['steps']['import_verify'] + c['steps']['auditor_init'] + c['steps']['first_audit']
                    for c in colds]
    lines.append(_row('  of which checker (import+init+audit)', summarize(checker_cold)))
    lines.append(_row('  (whole child process, wall)', summarize([c['process_wall_seconds'] for c in colds])))
    second = [c['steps']['second_audit'] for c in colds]
    lines.append(_row('same audit again, same process', summarize(second)))
    if not any('first_generate' in c['steps'] for c in colds):
        lines.append('  (no live generate in cold runs: --generate 0)')

    s = build['steps']
    lines += ['', 'INDEX', '-' * 40,
              f"  build from empty ({build['chunks']} chunks, {len(build['added'])} PDFs): "
              f"{_ms(1000 * s['build_from_empty'])} + {_ms(1000 * s['load_documents'])} PDF text extraction",
              f"  no-op sync on an unchanged corpus: {_ms(1000 * s['noop_sync'])}"]

    samples = warm['samples']
    audits = samples['audit']
    lines += ['', f"WARM - models loaded, {WARMUP_CALLS} untimed warm-up calls, {meta['reps']} reps "
                  f"over the 60 in-domain questions / saved answers", '-' * 40, header,
              _row('retrieve (embed query + Chroma)', summarize([x['seconds'] for x in samples['retrieve']])),
              _row('audit (segment + NLI)', summarize([x['seconds'] for x in audits])),
              _row('  of which spaCy segmentation', summarize([x['segment_seconds'] for x in audits]))]
    pairs = np.array([x['pairs'] for x in audits], dtype=float)
    secs = np.array([x['seconds'] - x['segment_seconds'] for x in audits])
    if len(pairs) and pairs.sum():
        slope, intercept = np.polyfit(pairs, secs * 1000.0, 1)
        lines.append(f"  NLI cost vs pairs (sentences x chunks, {int(pairs.min())}-{int(pairs.max())}): "
                     f"~{slope:.1f} ms per pair + {intercept:.1f} ms fixed (least squares)")
    for lo, hi in ((1, 3), (4, 12), (13, 36)):
        bucket = [x['seconds'] for x in audits if lo <= x['pairs'] <= hi]
        if bucket:
            lines.append(_row(f'  audit, {lo}-{hi} pairs', summarize(bucket)))

    idle = samples.get('idle') or []
    if idle:
        pauses = sorted({x['pause'] for x in idle})
        lines += ['', f"IDLE - the same 4-sentence answer, back-to-back vs after an idle pause "
                      f"(real users pause between questions)", '-' * 40, header]
        for pause in pauses:
            label = 'audit, back-to-back' if pause == 0 else f'audit after {pause:g} s idle'
            lines.append(_row(label, summarize([x['seconds'] for x in idle if x['pause'] == pause])))

    layers = samples.get('layers') or []
    if layers:
        plain = [x['plain_generate'] for x in layers]
        rag = [x['retrieve'] + x['rag_generate'] for x in layers]
        checked = [r + x['audit'] for r, x in zip(rag, layers)]
        lines += ['', f"WHAT EACH LAYER ADDS - {len(layers)} live questions, each asked plain and with RAG "
                      f"(random order), Groq calls paced {meta['pace_seconds']:g} s apart", '-' * 40, header,
                  _row('1. plain Groq, question only', summarize(plain)),
                  _row('2. RAG: retrieve + Groq with chunks', summarize(rag)),
                  _row('3. RAG + RagShield checker (the app)', summarize(checked)),
                  '', '  added, per question (paired differences):', header,
                  _row('RAG over plain Groq  (2 - 1)', summarize([r - p for r, p in zip(rag, plain)])),
                  _row('  of which retrieve', summarize([x['retrieve'] for x in layers])),
                  _row('checker over RAG     (3 - 2)', summarize([x['audit'] for x in layers])),
                  f"  checker share of the app's total: median {100 * np.median([x['audit'] / c for x, c in zip(layers, checked)]):.0f}% "
                  f"per question, {100 * sum(x['audit'] for x in layers) / sum(checked):.0f}% of all time",
                  f"  tokens, median plain vs RAG: prompt {_median_tokens(layers, 'plain_prompt_tokens')} vs "
                  f"{_median_tokens(layers, 'rag_prompt_tokens')}, completion (includes reasoning) "
                  f"{_median_tokens(layers, 'plain_completion_tokens')} vs {_median_tokens(layers, 'rag_completion_tokens')}",
                  f"  RAG answers {int(np.median([x['sentences'] for x in layers]))} sentences median; "
                  f"retried after back-off, excluded above: {warm['retries']['rate_limited']} rate limits (429), "
                  f"{warm['retries']['connection_errors']} connection errors (the app would show an error); "
                  f"pacing added {warm['paced_seconds']:.0f} s."]
    return '\n'.join(lines) + '\n'


def _median_tokens(rows, key):
    values = [x[key] for x in rows if x.get(key) is not None]
    return f"{np.median(values):.0f}" if values else 'n/a'


def report(out_dir):
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / 'run.json').read_text(encoding='utf-8'))
    build = json.loads((out_dir / 'build.json').read_text(encoding='utf-8'))
    colds = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(out_dir.glob('cold*.json'))]
    warm = json.loads((out_dir / 'warm.json').read_text(encoding='utf-8'))
    text = build_report(meta, build, colds, warm)
    (out_dir / 'report.txt').write_text(text, encoding='utf-8')
    return text


# --------------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m eval.latency')
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run', help='measure (spawns one child process per measurement)')
    r.add_argument('--device', choices=('auto', 'cpu'), default='auto')
    r.add_argument('--cold-runs', type=int, default=3)
    r.add_argument('--reps', type=int, default=3)
    r.add_argument('--generate', type=int, default=20, help='live questions for the layer comparison, 2 Groq calls each (0 for none)')
    r.add_argument('--pace', type=float, default=9.0, help='seconds between live query starts')
    r.add_argument('--name', help='results subdirectory (default: the device)')
    p = sub.add_parser('report', help='re-render a saved run')
    p.add_argument('results_dir')

    for hidden in ('_build', '_cold', '_warm'):
        h = sub.add_parser(hidden)
        h.add_argument('--index', required=True)
        h.add_argument('--out', required=True)
        h.add_argument('--generate', type=int, default=0)
        h.add_argument('--reps', type=int, default=3)
        h.add_argument('--pace', type=float, default=9.0)
        h.add_argument('--seed', type=int, default=0)

    args = parser.parse_args(argv)
    if args.command == 'run':
        text = run(args.device, args.cold_runs, args.reps, args.generate, args.pace,
                   args.name or args.device)
    elif args.command == 'report':
        text = report(args.results_dir)
    elif args.command == '_build':
        return child_build(Path(args.index), args.out)
    elif args.command == '_cold':
        return child_cold(Path(args.index), args.out, bool(args.generate))
    else:
        return child_warm(Path(args.index), args.out, args.reps, args.generate, args.pace, args.seed)
    sys.stdout.reconfigure(encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
