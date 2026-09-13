"""RAGTruth, mapped from character-span annotations to sentence labels.

RAGTruth annotates hallucinations as character spans over each response. The
auditor does not score spans; it scores the sentences `verify.segmenter` splits
a response into. This module projects one onto the other: a sentence is
**unsupported** if any annotated span overlaps it.

Nothing here is committed
-------------------------
Neither the raw dataset nor the derived labels live in the repo. The raw
parquet is fetched from the HuggingFace hub at a pinned revision into the HF
cache; the labels are generated locally into `eval/data/ragtruth/` (gitignored)
as gzipped JSONL of sentence offsets and labels, no text. Every input is
pinned -- dataset revision, spaCy, the spaCy model -- and the build is
byte-reproducible, so this script plus those pins *is* the dataset. A run that
reports numbers should quote the label file's SHA-256 from `manifest.json`, so
anyone rebuilding can confirm they got the same labels.

Text is joined back from the raw rows at load time, and every response carries
a SHA-256 of the output it was labelled against, so a changed upstream row
fails loudly instead of silently misaligning offsets. The manifest records a
SHA-256 of each label file too, checked on load.

The labels also depend on segmentation -- a different spaCy model, or any edit
to `verify/segmenter.py`, can split a response differently and move every
offset. `manifest.json` records the spaCy and model versions plus a hash of the
segmenter source, and loading refuses a mismatch.

Rebuild with `python -m eval.datasets ragtruth build` (~12 minutes of spaCy
for both splits); `... ragtruth spotcheck` prints labelled responses with
their raw spans bracketed.

Boundary grazes
---------------
spaCy attaches a list marker to the end of the sentence before it ("...high
heat.\\n8."), and annotators routinely start a span at that marker. Taken
literally, "any overlap" then marks the previous sentence unsupported because
its last two characters are "8.". On the test split this affected 18 spans,
every one of them a list marker, a bullet, or a lone punctuation mark.

So when a span touches more than one sentence, touches whose overlapping text
is only punctuation, or a list marker ("8.", "3)") opening a new line, are
ignored -- counted in `grazes_ignored`, never dropped silently. The newline
condition is what keeps "...is 22." at the end of a sentence a claim. A span
that touches a single sentence always marks it, whatever the overlap, and a
span whose every touch is a graze keeps them all rather than vanish.

Spans that overlap no kept sentence at all (they fall wholly inside a dropped
<=5-character fragment) are recorded per response in `unmapped_spans`.
"""
import argparse
import gzip
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ID = 'wandb/RAGTruth-processed'
# Pinned: the labels below are offsets into these exact rows.
REVISION = 'eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b'
SPLITS = ('train', 'test')
TASK_TYPES = ('QA', 'Summary', 'Data2txt')

DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'ragtruth'

# Overlap text that carries no claim. Applied only to spans touching 2+
# sentences. Punctuation and symbols never do; a number followed by "." or ")"
# only does when it opens a new line, which is what a list marker looks like --
# "...is 22." at the end of a sentence is a claim, not a marker.
_PUNCTUATION_ONLY = re.compile(r'[\W_]*')
_LIST_MARKER = re.compile(r'\d{1,3}[.)]')


# ------------------------------------------------------------ raw dataset

def raw_path(split):
    """Local path of a raw split, downloading it into the HF cache if needed.

    Honours HF_HOME. On this machine that must point at D:, not C:.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO_ID, f'data/{split}-00000-of-00001.parquet',
                           repo_type='dataset', revision=REVISION)


def load_raw(split):
    import pyarrow.parquet as pq
    return pq.read_table(raw_path(split)).to_pylist()


def output_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# ------------------------------------------------------- span -> sentence

def _is_graze(text, lo, hi):
    """True if `text[lo:hi]`, the part of a span inside one sentence, is no claim."""
    overlap = text[lo:hi].strip()
    if _PUNCTUATION_ONLY.fullmatch(overlap):
        return True
    if _LIST_MARKER.fullmatch(overlap):
        preceding = text[:lo + text[lo:hi].index(overlap)]
        gap = preceding[len(preceding.rstrip()):]
        return '\n' in gap or not preceding.strip()
    return False


def map_spans_to_sentences(text, sentence_spans, hallucination_spans):
    """Which annotated spans land on which sentences. Pure; no model needed.

    `sentence_spans` is `[(start, end, sentence), ...]` as returned by
    `split_into_sentence_spans`; `hallucination_spans` are RAGTruth span dicts
    with at least `start` and `end`.

    Returns `(hits, grazes_ignored, unmapped)`: `hits[i]` lists the indices of
    spans marking sentence i, and `unmapped` lists the indices of spans that
    overlap no sentence. Every span index appears in `hits` or in `unmapped`.
    """
    hits = [[] for _ in sentence_spans]
    grazes_ignored = 0
    unmapped = []

    for j, span in enumerate(hallucination_spans):
        touched = []
        for i, (start, end, _) in enumerate(sentence_spans):
            lo, hi = max(span['start'], start), min(span['end'], end)
            if hi > lo:
                touched.append((i, lo, hi))

        if not touched:
            unmapped.append(j)
            continue

        if len(touched) > 1:
            substantive = [t for t in touched if not _is_graze(text, t[1], t[2])]
            # Only ever narrows: if every touch is a graze, keep them all
            # rather than lose the span.
            if substantive:
                grazes_ignored += len(touched) - len(substantive)
                touched = substantive

        for i, _, _ in touched:
            hits[i].append(j)

    return hits, grazes_ignored, unmapped


def label_response(row, segment=None):
    """Sentence labels for one raw RAGTruth row, as a JSON-ready record."""
    if segment is None:
        from verify.segmenter import split_into_sentence_spans as segment

    text = row['output']
    spans = json.loads(row['hallucination_labels'])
    sentence_spans = segment(text)
    hits, grazes, unmapped = map_spans_to_sentences(text, sentence_spans, spans)

    sentences = []
    for (start, end, sentence), marks in zip(sentence_spans, hits):
        assert text[start:end] == sentence, (row['id'], start, end)
        marking = [spans[j] for j in marks]
        sentences.append({
            'start': start,
            'end': end,
            'unsupported': bool(marking),
            'label_types': sorted({s['label_type'] for s in marking}),
            # Kept rather than filtered: whether "true but not in the source"
            # and null-field spans count as hallucinations is a reporting
            # choice, and it should be made visibly at report time.
            'implicit_true': any(s.get('implicit_true') for s in marking),
            'due_to_null': any(s.get('due_to_null') for s in marking),
        })

    mapped = {j for marks in hits for j in marks}
    # The coverage guarantee: no span is lost without being recorded.
    assert len(mapped) + len(unmapped) == len(spans), row['id']

    return {
        'id': row['id'],
        'task_type': row['task_type'],
        'model': row['model'],
        'quality': row['quality'],
        'output_sha256': output_sha256(text),
        'n_spans': len(spans),
        'grazes_ignored': grazes,
        'unmapped_spans': [{'start': spans[j]['start'], 'end': spans[j]['end'],
                            'text': spans[j]['text']} for j in unmapped],
        'sentences': sentences,
    }


# ------------------------------------------------------------ build + load

def _source_sha256(data):
    """SHA-256 of source bytes with line endings normalised.

    Normalised because git converts to CRLF on checkout here and not on Linux,
    and a fingerprint that differs per platform would refuse valid labels.
    """
    return hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()


def _segmenter_versions():
    """Everything the sentence offsets depend on.

    Versions alone are not enough: a change to how `verify/segmenter.py` splits
    or filters moves offsets without moving any version number, so the source
    is fingerprinted too. A comment edit will also demand a rebuild; that false
    alarm is cheap, and the silent misalignment it prevents is not.
    """
    import spacy
    from verify import segmenter
    return {
        'spacy': spacy.__version__,
        'model': f"{segmenter.nlp.meta['name']}-{segmenter.nlp.meta['version']}",
        'min_sentence_chars': segmenter.MIN_SENTENCE_CHARS,
        'source_sha256': _source_sha256(Path(segmenter.__file__).read_bytes()),
    }


def summarize(records):
    """Counts per task type, including how many responses are *mixed*.

    Mixed responses are the only ones within-response AUROC can use, so this is
    the number that says how much evidence the headline metric rests on.
    """
    out = {}
    for rec in records:
        s = out.setdefault(rec['task_type'], {
            'responses': 0, 'sentences': 0, 'unsupported': 0,
            'mixed_responses': 0, 'spans': 0, 'unmapped_spans': 0,
            'grazes_ignored': 0,
        })
        labels = [x['unsupported'] for x in rec['sentences']]
        s['responses'] += 1
        s['sentences'] += len(labels)
        s['unsupported'] += sum(labels)
        s['mixed_responses'] += 0 < sum(labels) < len(labels)
        s['spans'] += rec['n_spans']
        s['unmapped_spans'] += len(rec['unmapped_spans'])
        s['grazes_ignored'] += rec['grazes_ignored']
    return dict(sorted(out.items()))


def labels_path(split, data_dir=DATA_DIR):
    return Path(data_dir) / f'{split}.jsonl.gz'


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_labels(records, path):
    """Gzipped JSONL, byte-for-byte reproducible.

    Gzip because the train labels are ~14 MB as plain JSONL and ~10x smaller
    compressed, and they are generated, never hand-edited, so diffability buys
    nothing. `mtime=0` and no embedded filename make the bytes a pure function
    of the records: rebuilding unchanged labels yields an identical file, which
    is the reproducibility check.
    """
    with open(path, 'wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
            for rec in records:
                line = json.dumps(rec, ensure_ascii=False, separators=(',', ':'))
                gz.write((line + '\n').encode('utf-8'))


def build(splits=SPLITS, out_dir=DATA_DIR, progress=True):
    """Label every response; write `{split}.jsonl.gz` and `manifest.json` (gitignored)."""
    from datetime import date

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / 'manifest.json'
    manifest = {
        'dataset': REPO_ID,
        'revision': REVISION,
        'segmenter': _segmenter_versions(),
        'built': date.today().isoformat(),
        'splits': {},
        'sha256': {},
    }
    if manifest_path.exists():
        # Keep the entries for splits not rebuilt this run.
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['splits'] = previous.get('splits', {})
        manifest['sha256'] = previous.get('sha256', {})

    for split in splits:
        rows = load_raw(split)
        records = []
        for n, row in enumerate(rows, 1):
            records.append(label_response(row))
            if progress and n % 500 == 0:
                print(f"  {split}: {n}/{len(rows)}", file=sys.stderr, flush=True)
        path = labels_path(split, out_dir)
        write_labels(records, path)
        manifest['splits'][split] = summarize(records)
        manifest['sha256'][split] = _file_sha256(path)

    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def load_labels(split, data_dir=DATA_DIR, check_segmenter=True):
    """The locally built sentence labels for a split, one record per response.

    Refuses a label file whose hash is not the one the manifest recorded, and
    labels built under a different segmenter than the installed one. Never
    builds implicitly: labelling takes ~12 minutes, which should not happen as
    a side effect of starting an eval run.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    path = labels_path(split, data_dir)
    manifest_path = Path(data_dir) / 'manifest.json'
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"no RAGTruth labels for split {split!r} in {data_dir}. They are "
            f"generated, not committed: run "
            f"`python -m eval.datasets ragtruth build --split {split}` (~12 min "
            f"for both splits; needs HF_HOME on a drive with ~30 MB free)."
        )
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if split not in manifest.get('sha256', {}):
        raise FileNotFoundError(
            f"manifest.json has no entry for split {split!r}; rebuild it with "
            f"`python -m eval.datasets ragtruth build --split {split}`."
        )
    if _file_sha256(path) != manifest['sha256'][split]:
        raise ValueError(f"{path} does not match the hash in manifest.json; "
                         f"rebuild rather than trust it")
    if check_segmenter:
        installed = _segmenter_versions()
        if manifest['segmenter'] != installed:
            raise RuntimeError(
                f"RAGTruth labels were built with segmenter {manifest['segmenter']} "
                f"but {installed} is installed. Sentence offsets would not match "
                f"what the auditor scores. Rebuild with "
                f"`python -m eval.datasets ragtruth build`."
            )
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        return [json.loads(line) for line in fh]


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int
    unsupported: bool
    label_types: tuple
    implicit_true: bool
    due_to_null: bool


@dataclass(frozen=True)
class Example:
    id: str
    task_type: str
    model: str
    quality: str
    query: str
    context: str
    output: str
    sentences: tuple


def join(labels, raw_rows):
    """Attach raw text to label records. Raises on any drift between them."""
    by_id = {row['id']: row for row in raw_rows}
    examples = []
    for rec in labels:
        row = by_id.get(rec['id'])
        if row is None:
            raise KeyError(f"response {rec['id']} is labelled but absent from the raw split")
        if output_sha256(row['output']) != rec['output_sha256']:
            raise ValueError(
                f"response {rec['id']}: raw output no longer matches the text it "
                f"was labelled against; offsets would be meaningless"
            )
        text = row['output']
        examples.append(Example(
            id=rec['id'], task_type=rec['task_type'], model=rec['model'],
            quality=rec['quality'], query=row['query'], context=row['context'],
            output=text,
            sentences=tuple(
                Sentence(text[s['start']:s['end']], s['start'], s['end'],
                         s['unsupported'], tuple(s['label_types']),
                         s['implicit_true'], s['due_to_null'])
                for s in rec['sentences']
            ),
        ))
    return examples


def load_examples(split, task_types=None):
    """Labelled examples with text, optionally restricted to some task types.

    Task-type selection is left to the caller on purpose (PLAN.md D3): the
    headline is QA, but all three are reported, so nothing is filtered here.
    """
    if task_types is not None:
        unknown = set(task_types) - set(TASK_TYPES)
        if unknown:
            raise ValueError(f"unknown task types {sorted(unknown)}; expected {TASK_TYPES}")
    labels = load_labels(split)
    if task_types is not None:
        labels = [rec for rec in labels if rec['task_type'] in task_types]
    return join(labels, load_raw(split))


# ------------------------------------------------------------------- CLI

def _render(example, raw_row):
    """A response with annotated spans bracketed, then its sentence labels."""
    spans = sorted(json.loads(raw_row['hallucination_labels']), key=lambda s: s['start'])
    text, pieces, cursor = example.output, [], 0
    for s in spans:
        pieces += [text[cursor:s['start']], '[[', text[s['start']:s['end']],
                   f"]]<{s['label_type']}>"]
        cursor = s['end']
    pieces.append(text[cursor:])
    lines = [f"=== {example.id}  {example.task_type}  {example.model}  "
             f"quality={example.quality}", ''.join(pieces), '--- sentences']
    for i, s in enumerate(example.sentences):
        mark = 'UNSUPPORTED' if s.unsupported else 'supported  '
        lines.append(f"  {i:>2} {mark} {s.text!r}")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eval.datasets ragtruth")
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build', help='label the raw splits and write eval/data/ragtruth')
    b.add_argument('--split', choices=SPLITS, nargs='+', default=list(SPLITS))
    c = sub.add_parser('spotcheck', help='print random labelled responses with their spans')
    c.add_argument('--split', choices=SPLITS, default='test')
    c.add_argument('--n', type=int, default=10)
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--hallucinated-only', action='store_true')
    args = parser.parse_args(argv)

    if args.command == 'build':
        manifest = build(args.split)
        print(json.dumps(manifest, indent=2))
        return

    raw = {row['id']: row for row in load_raw(args.split)}
    examples = join(load_labels(args.split), raw.values())
    if args.hallucinated_only:
        examples = [e for e in examples if any(s.unsupported for s in e.sentences)]
    for example in random.Random(args.seed).sample(examples, args.n):
        print(_render(example, raw[example.id]) + '\n')
