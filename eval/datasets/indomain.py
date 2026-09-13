"""E5: a small labelled set over the app's own demo PDFs.

    python -m eval.datasets indomain collect   # retrieve + generate; never audits
    python -m eval.datasets indomain sheet     # the blind labelling sheet
    python -m eval.datasets indomain check     # validate labels against answers

RAGTruth measures the verifier on MS MARCO / CNN-DM / Yelp text with no
retrieval step. This set measures the pipeline the demo actually runs: PDF
extraction, chunking, top-k retrieval, Groq generation, then the auditor.

Everything in `eval/data/indomain/` is committed, unlike RAGTruth's labels.
None of it is derived from something pinned: the questions and labels are
written by hand, and the answers come from a hosted LLM that will not
reproduce them, so the files *are* the dataset (PLAN.md D4).

Files
-----
questions.jsonl        written before anything was generated
answers.jsonl          `collect` output: retrieved chunks, answer, sentence offsets
collection.json        how the answers were produced (models, config, doc hashes)
sentence_labels.jsonl  one label per answer sentence, with annotator and a note
question_labels.jsonl  per question: did retrieval surface the answer at all
perturbations.jsonl    supported sentences edited to be unsupported

Labels are relative to the **retrieved chunks** - the evidence the auditor
sees - not to the whole PDF (PLAN.md D4 lists the definitions).
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'eval' / 'data' / 'indomain'
DOCS_DIR = REPO_ROOT / 'demo' / 'example_docs'
INDEX_DIR = REPO_ROOT / 'eval' / 'results' / 'indomain-chroma'

FILES = {
    'questions': 'questions.jsonl',
    'answers': 'answers.jsonl',
    'sentence_labels': 'sentence_labels.jsonl',
    'question_labels': 'question_labels.jsonl',
    'perturbations': 'perturbations.jsonl',
}

# SUPPORTED / UNSUPPORTED are scored; META and NO_CLAIM carry no claim about
# the world the source could support, so they are reported separately (D2).
SCORED_LABELS = ('SUPPORTED', 'UNSUPPORTED')
LABELS = SCORED_LABELS + ('META', 'NO_CLAIM')
PERTURBATION_KINDS = ('number', 'negation', 'entity')


# ------------------------------------------------------------------ files

def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(rows, path):
    """One JSON object per line, UTF-8, LF: the files are committed and diffed."""
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')


def sentence_id(question_id, index):
    return f'{question_id}-s{index}'


def load_raw(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    return {key: read_jsonl(data_dir / name) for key, name in FILES.items()}


# ------------------------------------------------------------- validation

def validate(raw):
    """Every problem with the labelled set, as a list of strings (empty = valid).

    Collects rather than raising on the first, so one pass shows every gap in a
    half-finished labelling job.
    """
    problems = []
    questions = {q['id']: q for q in raw['questions']}
    answers = {a['question_id']: a for a in raw['answers']}
    if len(questions) != len(raw['questions']):
        problems.append('duplicate question ids')

    for qid in questions:
        if qid not in answers:
            problems.append(f'{qid}: no collected answer')
    for qid in answers:
        if qid not in questions:
            problems.append(f'{qid}: answer for an unknown question')

    sentences = {s['id']: s for a in raw['answers'] for s in a['sentences']}
    for a in raw['answers']:
        for i, s in enumerate(a['sentences']):
            if s['id'] != sentence_id(a['question_id'], i):
                problems.append(f"{s['id']}: sentence id out of order")
            if a['answer'][s['start']:s['end']] != s['text']:
                problems.append(f"{s['id']}: offsets do not reproduce the text")

    seen = {}
    for row in raw['sentence_labels']:
        sid = row.get('sentence_id')
        if sid not in sentences:
            problems.append(f'{sid}: label for an unknown sentence')
        if sid in seen:
            problems.append(f'{sid}: labelled twice')
        seen[sid] = row.get('label')
        if row.get('label') not in LABELS:
            problems.append(f"{sid}: label {row.get('label')!r} not in {LABELS}")
        if not row.get('annotator'):
            problems.append(f'{sid}: no annotator')
        for flag in ('borderline', 'multi_chunk'):
            if not isinstance(row.get(flag, False), bool):
                problems.append(f'{sid}: {flag} must be true or false')
        if row.get('multi_chunk') and row.get('label') != 'SUPPORTED':
            problems.append(f'{sid}: multi_chunk only applies to SUPPORTED sentences')
    for sid in sentences:
        if sid not in seen:
            problems.append(f'{sid}: unlabelled')

    labelled_questions = set()
    for row in raw['question_labels']:
        qid = row.get('question_id')
        if qid not in questions:
            problems.append(f'{qid}: question label for an unknown question')
        if qid in labelled_questions:
            problems.append(f'{qid}: question labelled twice')
        labelled_questions.add(qid)
        if not isinstance(row.get('retrieval_has_answer'), bool):
            problems.append(f'{qid}: retrieval_has_answer must be true or false')
    for qid in questions:
        if qid not in labelled_questions:
            problems.append(f'{qid}: no question label')

    ids = set()
    for row in raw['perturbations']:
        pid, source = row.get('id'), row.get('source')
        if pid in ids or pid in sentences:
            problems.append(f'{pid}: duplicate perturbation id')
        ids.add(pid)
        if source not in sentences:
            problems.append(f'{pid}: source {source!r} is not an answer sentence')
        elif seen.get(source) != 'SUPPORTED':
            problems.append(f'{pid}: source {source} is labelled {seen.get(source)}, not SUPPORTED')
        elif row.get('sentence', '').strip() == sentences[source]['text'].strip():
            problems.append(f'{pid}: perturbation is identical to its source')
        if row.get('kind') not in PERTURBATION_KINDS:
            problems.append(f"{pid}: kind {row.get('kind')!r} not in {PERTURBATION_KINDS}")
        if not row.get('annotator'):
            problems.append(f'{pid}: no annotator')
    return problems


def segmentation_drift(answers, segment=None):
    """Answer sentences the current segmenter would no longer produce.

    The labels are attached to saved offsets. If `verify/segmenter.py` or the
    spaCy model changes, the auditor would score different sentences from the
    labelled ones, so scoring refuses to run until this is empty.
    """
    if segment is None:
        from verify.segmenter import split_into_sentence_spans as segment
    drift = []
    for a in answers:
        now = [(start, end, text) for start, end, text in segment(a['answer'])]
        saved = [(s['start'], s['end'], s['text']) for s in a['sentences']]
        if now != saved:
            drift.append(a['question_id'])
    return drift


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_fingerprint(data_dir=DATA_DIR):
    """SHA-256 over the committed label files, quoted with every E5 number."""
    digest = hashlib.sha256()
    for key in ('sentence_labels', 'question_labels', 'perturbations'):
        path = Path(data_dir) / FILES[key]
        digest.update(key.encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n') if path.exists() else b'')
    return digest.hexdigest()


def load(data_dir=DATA_DIR, check_segmenter=True):
    """The validated set; raises if anything is missing, unlabelled or drifted."""
    raw = load_raw(data_dir)
    if not raw['answers']:
        raise FileNotFoundError(
            f"no answers in {Path(data_dir) / FILES['answers']}; "
            f"run `python -m eval.datasets indomain collect`")
    problems = validate(raw)
    if problems:
        shown = '\n  '.join(problems[:20])
        more = f'\n  ... and {len(problems) - 20} more' if len(problems) > 20 else ''
        raise ValueError(f'in-domain set is not valid ({len(problems)} problems):\n  {shown}{more}')
    if check_segmenter:
        drift = segmentation_drift(raw['answers'])
        if drift:
            raise ValueError(
                f"the segmenter no longer reproduces the labelled sentences for {drift}; "
                f"the labels would be attached to different text")
    return raw


# ------------------------------------------------------------- collection

def _git_state():
    import subprocess

    def git(*args):
        return subprocess.run(['git', *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    return {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain'))}


def _generate_with_backoff(generator, question, context, attempts=8):
    """One answer, waiting out Groq's per-minute token limit instead of failing.

    The free tier allows 8,000 tokens a minute and one call is ~1,100, so a
    30-question run trips the limit several times.
    """
    import time

    import groq

    for attempt in range(attempts):
        try:
            return generator.generate_answer(question, context)
        except groq.RateLimitError:
            if attempt == attempts - 1:
                raise
            wait = 15 * (attempt + 1)
            print(f"    rate limited; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)


def collect(data_dir=DATA_DIR, index_dir=INDEX_DIR, question_ids=None):
    """Run retrieval and generation for each question. The auditor is not loaded.

    Uses a private Chroma index over `demo/example_docs/` only, so uploads in
    the app's own `chroma_db/` cannot leak into the retrieved evidence.

    **Never regenerates an answer that already exists** unless its id is named
    explicitly: labels are attached to the saved text, and Groq will not
    reproduce it. Answers are saved after every question, so an interrupted run
    resumes where it stopped.
    """
    from ingest import DocumentLoader
    from index import RagShieldIndex, cfg
    from rag import RagGenerator, Retriever
    from verify.segmenter import split_into_sentence_spans

    data_dir = Path(data_dir)
    all_questions = read_jsonl(data_dir / FILES['questions'])
    order = [q['id'] for q in all_questions]
    existing = {a['question_id']: a for a in read_jsonl(data_dir / FILES['answers'])}
    if question_ids:
        questions = [q for q in all_questions if q['id'] in set(question_ids)]
    else:
        questions = [q for q in all_questions if q['id'] not in existing]

    index = RagShieldIndex(db_path=str(index_dir))
    docs = DocumentLoader(str(DOCS_DIR)).load()
    index.sync_documents(docs)
    retriever = Retriever(index=index)
    generator = RagGenerator()

    for n, q in enumerate(questions, 1):
        chunks = retriever.retrieve(q['question'])
        context = '\n\n'.join(c['text'] for c in chunks)  # as app.py builds it
        answer = _generate_with_backoff(generator, q['question'], context)
        spans = split_into_sentence_spans(answer)
        existing[q['id']] = {
            'question_id': q['id'],
            'retrieved': [{'chunk_id': c['chunk_id'], 'doc_id': c['doc_id'],
                           'distance': c['distance'], 'text': c['text']} for c in chunks],
            'answer': answer,
            'sentences': [{'id': sentence_id(q['id'], i), 'start': start, 'end': end,
                           'text': text} for i, (start, end, text) in enumerate(spans)],
        }
        write_jsonl([existing[qid] for qid in order if qid in existing],
                    data_dir / FILES['answers'])
        print(f"  {n}/{len(questions)} {q['id']}: {len(spans)} sentences", file=sys.stderr)

    from eval.datasets.ragtruth import _segmenter_versions
    meta = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'git': _git_state(),
        'generator': {'model': cfg['models']['generator'], 'temperature': 0.1,
                      'prompt': 'rag/generator.py RagGenerator.generate_answer'},
        'embeddings': cfg['models']['embeddings'],
        'retrieval': dict(cfg['retrieval']),
        'ingestion': dict(cfg['ingestion']),
        'documents': {d.name: sha256_of(d) for d in sorted(DOCS_DIR.iterdir()) if d.is_file()},
        'segmenter': _segmenter_versions(),
    }
    (data_dir / 'collection.json').write_text(json.dumps(meta, indent=2) + '\n',
                                              encoding='utf-8', newline='\n')
    return meta


# ------------------------------------------------------------------ sheet

def sheet(raw):
    """The labelling view: question, retrieved chunks, numbered sentences.

    Deliberately shows no verdict or score - the labels must be written before
    the auditor has looked at this set.
    """
    questions = {q['id']: q for q in raw['questions']}
    lines = []
    for a in raw['answers']:
        q = questions[a['question_id']]
        lines += ['=' * 78, f"{q['id']}  [{q['doc']}]  answerable in doc: {q['answerable']}",
                  f"Q: {q['question']}", '']
        for i, c in enumerate(a['retrieved']):
            lines += [f"--- chunk {i}: {c['chunk_id']}", c['text'], '']
        lines.append('--- answer sentences')
        for s in a['sentences']:
            lines.append(f"[{s['id']}] {s['text']}")
        lines.append('')
    return '\n'.join(lines)


# -------------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m eval.datasets indomain')
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect', help='retrieve and generate an answer per question (calls Groq)')
    c.add_argument('--questions', nargs='+', help='only these question ids')
    s = sub.add_parser('sheet', help='print the blind labelling sheet')
    s.add_argument('--out', help='write to this file instead of stdout')
    sub.add_parser('check', help='validate the labelled set')
    args = parser.parse_args(argv)

    if args.command == 'collect':
        collect(question_ids=args.questions)
        raw = load_raw()
        print(f"{len(raw['answers'])} answers, "
              f"{sum(len(a['sentences']) for a in raw['answers'])} sentences")
    elif args.command == 'sheet':
        text = sheet(load_raw())
        if args.out:
            Path(args.out).write_text(text, encoding='utf-8')
        else:
            sys.stdout.reconfigure(encoding='utf-8')
            print(text)
    else:
        raw = load_raw()
        problems = validate(raw)
        drift = segmentation_drift(raw['answers'])
        for p in problems:
            print(p)
        if drift:
            print(f'segmentation drift: {drift}')
        n = sum(len(a['sentences']) for a in raw['answers'])
        print(f"{len(raw['answers'])} answers, {n} sentences, "
              f"{len(raw['sentence_labels'])} labels, {len(raw['perturbations'])} perturbations; "
              f"{len(problems)} problems, label sha256 {label_fingerprint()}")
        sys.exit(1 if problems or drift else 0)
