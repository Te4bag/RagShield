"""The E5 in-domain set: validation of the committed files, and the scorer.

Offline: no Groq, no weights, no spaCy segmentation (a stub segmenter and a
stub auditor stand in). What is pinned is what keeps the labels meaningful -
every sentence labelled exactly once, perturbations built only from supported
sentences, drift in segmentation or in the label files refusing to score - and
the rules the report applies.
"""
import json

import pytest

from eval import indomain as report
from eval.datasets import indomain


CHUNKS = [{'chunk_id': 'doc.pdf_ch0', 'doc_id': 'doc.pdf', 'distance': 0.1,
           'text': 'The encoder has six layers.'},
          {'chunk_id': 'doc.pdf_ch1', 'doc_id': 'doc.pdf', 'distance': 0.2,
           'text': 'Training took twelve hours.'}]


def answer(qid, sentences):
    text, spans = '', []
    for i, s in enumerate(sentences):
        if text:
            text += ' '
        start = len(text)
        text += s
        spans.append({'id': f'{qid}-s{i}', 'start': start, 'end': len(text), 'text': s})
    return {'question_id': qid, 'retrieved': CHUNKS, 'answer': text, 'sentences': spans}


def label(sid, value='SUPPORTED', **extra):
    return {'sentence_id': sid, 'label': value, 'annotator': 'tester', **extra}


def fixture():
    return {
        'questions': [
            {'id': 'q1', 'batch': 1, 'doc': 'doc.pdf', 'answerable': True, 'question': 'How many layers?'},
            {'id': 'q2', 'batch': 1, 'doc': 'doc.pdf', 'answerable': False, 'question': 'What licence?'},
        ],
        'answers': [
            answer('q1', ['The encoder has six layers.', 'It was trained on Mars.']),
            answer('q2', ["I don't know."]),
        ],
        'sentence_labels': [
            label('q1-s0'), label('q1-s1', 'UNSUPPORTED', borderline=True), label('q2-s0', 'META'),
        ],
        'question_labels': [
            {'question_id': 'q1', 'retrieval_has_answer': True},
            {'question_id': 'q2', 'retrieval_has_answer': False},
        ],
        'perturbations': [
            {'id': 'q1-s0-p', 'source': 'q1-s0', 'kind': 'number',
             'sentence': 'The encoder has eight layers.', 'annotator': 'tester'},
        ],
    }


# -------------------------------------------------------------- validation

def test_a_complete_set_has_no_problems():
    assert indomain.validate(fixture()) == []


def test_an_unlabelled_sentence_is_reported():
    raw = fixture()
    raw['sentence_labels'].pop()

    assert 'q2-s0: unlabelled' in indomain.validate(raw)


def test_a_sentence_labelled_twice_is_reported():
    raw = fixture()
    raw['sentence_labels'].append(label('q1-s0', 'UNSUPPORTED'))

    assert 'q1-s0: labelled twice' in indomain.validate(raw)


def test_an_unknown_label_value_is_reported():
    raw = fixture()
    raw['sentence_labels'][0]['label'] = 'HALLUCINATED'

    assert any("'HALLUCINATED' not in" in p for p in indomain.validate(raw))


def test_offsets_must_reproduce_the_sentence_text():
    """Labels are attached to offsets; if they point at other text, every label
    describes the wrong sentence."""
    raw = fixture()
    raw['answers'][0]['sentences'][1]['start'] += 1

    assert 'q1-s1: offsets do not reproduce the text' in indomain.validate(raw)


def test_a_perturbation_must_come_from_a_supported_sentence():
    """Editing an already-unsupported sentence proves nothing about the flag."""
    raw = fixture()
    raw['perturbations'][0]['source'] = 'q1-s1'

    assert any('labelled UNSUPPORTED, not SUPPORTED' in p for p in indomain.validate(raw))


def test_a_perturbation_identical_to_its_source_is_reported():
    raw = fixture()
    raw['perturbations'][0]['sentence'] = 'The encoder has six layers.'

    assert 'q1-s0-p: perturbation is identical to its source' in indomain.validate(raw)


def test_every_question_needs_a_retrieval_label():
    raw = fixture()
    raw['question_labels'].pop()

    assert 'q2: no question label' in indomain.validate(raw)


def test_multi_chunk_is_only_meaningful_for_supported_sentences():
    raw = fixture()
    raw['sentence_labels'][1]['multi_chunk'] = True

    assert 'q1-s1: multi_chunk only applies to SUPPORTED sentences' in indomain.validate(raw)


def test_segmentation_drift_names_the_answers_that_moved():
    raw = fixture()

    def segment(text):
        # Splits q1 as one sentence instead of two.
        return [(0, len(text), text)]

    assert indomain.segmentation_drift(raw['answers'], segment=segment) == ['q1']


def test_load_refuses_an_invalid_set(tmp_path):
    raw = fixture()
    raw['sentence_labels'].pop()
    for key, name in indomain.FILES.items():
        indomain.write_jsonl(raw[key], tmp_path / name)

    with pytest.raises(ValueError, match='q2-s0: unlabelled'):
        indomain.load(tmp_path, check_segmenter=False)


def test_load_names_the_collect_command_when_there_are_no_answers(tmp_path):
    indomain.write_jsonl(fixture()['questions'], tmp_path / indomain.FILES['questions'])

    with pytest.raises(FileNotFoundError, match='indomain collect'):
        indomain.load(tmp_path, check_segmenter=False)


def test_label_fingerprint_moves_with_any_label_and_ignores_line_endings(tmp_path):
    raw = fixture()
    for key, name in indomain.FILES.items():
        indomain.write_jsonl(raw[key], tmp_path / name)
    before = indomain.label_fingerprint(tmp_path)

    path = tmp_path / indomain.FILES['sentence_labels']
    path.write_bytes(path.read_bytes().replace(b'\n', b'\r\n'))
    assert indomain.label_fingerprint(tmp_path) == before

    raw['sentence_labels'][0]['label'] = 'UNSUPPORTED'
    indomain.write_jsonl(raw['sentence_labels'], path)
    assert indomain.label_fingerprint(tmp_path) != before


def test_the_labelling_sheet_shows_evidence_but_no_scores():
    text = indomain.sheet(fixture())

    assert '[q1-s1] It was trained on Mars.' in text
    assert 'The encoder has six layers.' in text
    for leak in ('ENTAILMENT', 'NEUTRAL', 'LOW_SUPPORT', 'P(entailment)', 'verdict'):
        assert leak not in text


# ----------------------------------------------------------------- scoring

class StubAuditor:
    """Splits on '. ' and reads P(entailment) from a table, keyed by sentence."""
    def __init__(self, p, green=0.85, floor=0.001):
        self.p, self.green, self.floor = p, green, floor
        self.premises_seen = []

    def audit_response(self, text, premises):
        self.premises_seen.append([c['chunk_id'] for c in premises])
        parts = [s if s.endswith('.') else s + '.' for s in text.split('. ')]
        out = []
        for s in parts:
            p = self.p[s]
            verdict = ('ENTAILMENT' if p >= self.green
                       else 'LOW_SUPPORT' if p < self.floor else 'NEUTRAL')
            out.append({'sentence': s, 'verdict': verdict, 'entailment': p,
                        'evidence': {'chunk_id': premises[0]['chunk_id']}})
        return out


P = {'The encoder has six layers.': 0.97, 'It was trained on Mars.': 0.0005,
     "I don't know.": 0.01, 'The encoder has eight layers.': 0.2}


def test_score_joins_labels_and_scores_perturbations_against_the_source_chunks():
    auditor = StubAuditor(P)

    records = report.score(fixture(), auditor)

    by_id = {r['id']: r for r in records}
    assert [r['id'] for r in records] == ['q1-s0', 'q1-s1', 'q2-s0', 'q1-s0-p']
    assert by_id['q1-s1']['label'] == 'UNSUPPORTED' and by_id['q1-s1']['borderline'] is True
    assert by_id['q1-s0-p']['label'] == 'UNSUPPORTED'
    assert by_id['q1-s0-p']['question_id'] == 'q1'
    assert by_id['q1-s0-p']['retrieval_has_answer'] is True
    assert auditor.premises_seen[-1] == ['doc.pdf_ch0', 'doc.pdf_ch1']


def test_score_refuses_when_the_auditor_segments_differently():
    raw = fixture()
    raw['answers'][0]['answer'] = raw['answers'][0]['answer'].replace('. ', '; ')
    auditor = StubAuditor({**P, 'The encoder has six layers; It was trained on Mars.': 0.5})

    with pytest.raises(report.AlignmentError, match='q1'):
        report.score(raw, auditor)


def test_answer_rule_counts_meta_sentences_toward_the_minimum():
    records = [
        {'origin': 'answer', 'question_id': 'a', 'label': 'SUPPORTED', 'p_entailment': 0.9},
        {'origin': 'answer', 'question_id': 'a', 'label': 'META', 'p_entailment': 0.0001},
        {'origin': 'answer', 'question_id': 'b', 'label': 'UNSUPPORTED', 'p_entailment': 0.5},
        {'origin': 'perturbed', 'question_id': 'b', 'label': 'UNSUPPORTED', 'p_entailment': 0.0},
    ]

    qids, gold, score, flagged = report.answer_level(records, floor=0.001)

    assert qids == ['a', 'b']
    assert gold.tolist() == [False, True]          # META is never the gold positive
    assert flagged.tolist() == [True, False]       # ...but it is underlined, so it counts
    assert score == pytest.approx([0.9999, 0.5])   # perturbations never enter answers


def test_rate_ci_is_the_share_within_the_subset():
    ci = report.rate_ci(['a', 'a', 'b', 'b'], [True, True, True, False], [True, False, True, True])

    assert ci.estimate == pytest.approx(2 / 3)


def _meta(labels_sha='abc'):
    return {'config': {'nli_model': 'stub', 'verification': {'entailment_threshold': 0.85,
                                                             'low_support_threshold': 0.001}},
            'collection': {'generator': {'model': 'stub-gen'}}, 'labels_sha256': labels_sha,
            'annotators': ['tester'], 'git': {'commit': 'deadbeef', 'dirty': False},
            'environment': {'device': 'cpu'}}


def test_report_carries_provenance_caveats_and_the_listed_errors():
    records = report.score(fixture(), StubAuditor(P))

    text = report.build_report(records, _meta())

    assert 'labels sha256 abc' in text
    assert 'annotator: tester' in text
    assert 'Never merge with RAGTruth' in text
    assert 'relative to the retrieved chunks' in text
    assert 'PERTURBATIONS' in text
    assert 'META sentences (D2) - 1' in text
    assert 'excluding 1 borderline labels' in text


def test_report_refuses_scores_from_different_labels(tmp_path, monkeypatch):
    (tmp_path / 'run.json').write_text(json.dumps(_meta('old')), encoding='utf-8')
    indomain.write_jsonl([], tmp_path / 'scores.jsonl')
    monkeypatch.setattr(indomain, 'label_fingerprint', lambda *a, **k: 'new')

    with pytest.raises(ValueError, match='re-score'):
        report.run_report(tmp_path)
