"""RAGTruth span-to-sentence mapping, on hand-built fixtures.

Offline: nothing here downloads the dataset or needs spaCy's output. The span
mapping is a pure function over offsets, and `label_response` takes an
injectable segmenter, so every sentence boundary below is written by hand.

One test does run the real segmenter, because it guards the contract the whole
module rests on: `split_into_sentence_spans` and `split_into_sentences` must
segment identically, or labels would describe sentences the auditor never sees.
"""
import json

import pytest

from eval.datasets import ragtruth as rt


def spans_of(text, *sentences):
    """(start, end, sentence) for sentences that occur in order in `text`."""
    out, cursor = [], 0
    for sentence in sentences:
        start = text.index(sentence, cursor)
        out.append((start, start + len(sentence), sentence))
        cursor = start + len(sentence)
    return out


def span(text, fragment, label_type='Evident Conflict', **flags):
    start = text.index(fragment)
    return {'start': start, 'end': start + len(fragment), 'text': fragment,
            'label_type': label_type, 'implicit_true': False,
            'due_to_null': False, **flags}


def row(text, spans, **fields):
    return {'id': 'r1', 'task_type': 'QA', 'model': 'm', 'quality': 'good',
            'output': text, 'hallucination_labels': json.dumps(spans), **fields}


# ------------------------------------------------------------- mapping

def test_a_span_inside_one_sentence_marks_only_that_sentence():
    text = "The sky is blue. The grass is purple. Water is wet."
    sentences = spans_of(text, "The sky is blue.", "The grass is purple.",
                         "Water is wet.")

    hits, grazes, unmapped = rt.map_spans_to_sentences(
        text, sentences, [span(text, "purple")])

    assert hits == [[], [0], []]
    assert (grazes, unmapped) == (0, [])


def test_a_span_crossing_a_boundary_marks_both_sentences():
    text = "Paris is in Spain. It has ten million people. Fine."
    sentences = spans_of(text, "Paris is in Spain.", "It has ten million people.",
                         "Fine.")

    hits, grazes, _ = rt.map_spans_to_sentences(
        text, sentences, [span(text, "Spain. It has ten million")])

    assert hits == [[0], [0], []]
    assert grazes == 0


def test_a_list_marker_graze_does_not_mark_the_previous_sentence():
    """The real RAGTruth pattern: spaCy ends sentence 7 with the "8." that opens
    item 8, and the annotator's span starts at that marker."""
    text = "7. Preheat the grill.\n8. Season the steak with salt."
    sentences = spans_of(text, "7. Preheat the grill.\n8.", "Season the steak with salt.")

    hits, grazes, unmapped = rt.map_spans_to_sentences(
        text, sentences, [span(text, "8. Season the steak with salt.")])

    assert hits == [[], [0]]
    assert grazes == 1
    assert unmapped == []


def test_a_punctuation_graze_does_not_mark_the_previous_sentence():
    text = "It is open daily. Delivery is free on weekends."
    sentences = spans_of(text, "It is open daily.", "Delivery is free on weekends.")
    # A span that begins on the previous sentence's full stop.
    start = text.index(". Delivery")
    bad = {'start': start, 'end': len(text), 'label_type': 'Evident Baseless Info',
           'text': text[start:]}

    hits, grazes, _ = rt.map_spans_to_sentences(text, sentences, [bad])

    assert hits == [[], [0]]
    assert grazes == 1


def test_a_number_ending_a_sentence_is_a_claim_not_a_list_marker():
    """No newline before it, so "22." is the hallucinated temperature itself."""
    text = "The temperature is 22. The humidity is 90 percent."
    sentences = spans_of(text, "The temperature is 22.", "The humidity is 90 percent.")

    hits, grazes, _ = rt.map_spans_to_sentences(
        text, sentences, [span(text, "22. The humidity is 90")])

    assert hits == [[0], [0]]
    assert grazes == 0


def test_a_short_overlap_inside_a_single_sentence_still_marks_it():
    """The graze rule applies only to spans touching 2+ sentences; a lone
    hallucinated number must never be discarded."""
    text = "The dose is 5. Take it daily."
    sentences = spans_of(text, "The dose is 5.", "Take it daily.")

    hits, grazes, _ = rt.map_spans_to_sentences(text, sentences, [span(text, "5.")])

    assert hits == [[0], []]
    assert grazes == 0


def test_a_span_that_only_grazes_keeps_its_touches_rather_than_vanish():
    text = "First one here.\n2.\n*Second one here."
    sentences = spans_of(text, "First one here.\n2.", "*Second one here.")
    # "2.\n*": a list marker on sentence 0, a bullet on sentence 1. Two touches,
    # both grazes -- ignoring both would erase the span entirely.
    start = text.index("2.\n*")
    marker_only = {'start': start, 'end': start + 4, 'label_type': 'x', 'text': '2.\n*'}

    hits, grazes, unmapped = rt.map_spans_to_sentences(text, sentences, [marker_only])

    assert hits == [[0], [0]]
    assert (grazes, unmapped) == (0, [])


def test_a_span_inside_a_dropped_fragment_is_recorded_as_unmapped():
    """The segmenter drops fragments of <=5 chars; a span living only there
    reaches no sentence and must be reported, not lost."""
    text = "A real sentence here. Ok. Another real sentence."
    sentences = spans_of(text, "A real sentence here.", "Another real sentence.")

    hits, _, unmapped = rt.map_spans_to_sentences(text, sentences, [span(text, "Ok.")])

    assert hits == [[], []]
    assert unmapped == [0]


def test_touching_offsets_do_not_count_as_overlap():
    text = "Alpha beta gamma.Delta epsilon zeta."
    sentences = spans_of(text, "Alpha beta gamma.", "Delta epsilon zeta.")
    boundary = text.index("Delta")

    hits, _, _ = rt.map_spans_to_sentences(
        text, sentences, [{'start': boundary, 'end': boundary + 5, 'label_type': 'x'}])

    assert hits == [[], [0]]


# ------------------------------------------------------- label_response

def test_label_response_builds_the_record():
    text = "Paris is the capital. It has 90 million people. It is in Europe."
    sentences = spans_of(text, "Paris is the capital.", "It has 90 million people.",
                         "It is in Europe.")
    spans = [span(text, "90 million", 'Evident Conflict'),
             span(text, "million people", 'Subtle Baseless Info', implicit_true=True)]

    rec = rt.label_response(row(text, spans), segment=lambda _: sentences)

    assert [s['unsupported'] for s in rec['sentences']] == [False, True, False]
    middle = rec['sentences'][1]
    assert middle['label_types'] == ['Evident Conflict', 'Subtle Baseless Info']
    assert middle['implicit_true'] is True
    assert middle['due_to_null'] is False
    assert (middle['start'], middle['end']) == sentences[1][:2]
    assert rec['n_spans'] == 2
    assert rec['output_sha256'] == rt.output_sha256(text)
    assert rec['task_type'] == 'QA'
    # Text is not stored; it is joined back from the raw row.
    assert 'text' not in middle


def test_label_response_rejects_offsets_that_do_not_slice_to_the_sentence():
    text = "One sentence here. Two sentence here."

    with pytest.raises(AssertionError):
        rt.label_response(row(text, []), segment=lambda _: [(0, 5, "One sentence here.")])


def test_summarize_counts_mixed_responses_per_task():
    def rec(task, labels):
        return {'task_type': task, 'n_spans': sum(labels), 'grazes_ignored': 0,
                'unmapped_spans': [],
                'sentences': [{'unsupported': x} for x in labels]}

    summary = rt.summarize([rec('QA', [True, False]), rec('QA', [False, False]),
                            rec('QA', [True, True]), rec('Summary', [False])])

    assert summary['QA'] == {'responses': 3, 'sentences': 6, 'unsupported': 3,
                             'mixed_responses': 1, 'spans': 3,
                             'unmapped_spans': 0, 'grazes_ignored': 0}
    assert summary['Summary']['mixed_responses'] == 0


# -------------------------------------------------------------- storage

RECORDS = [{'id': '1', 'sentences': [{'start': 0, 'end': 9, 'unsupported': True}]},
           {'id': '2', 'sentences': []}]


def _store(tmp_path, records=RECORDS, split='test'):
    path = rt.labels_path(split, tmp_path)
    rt.write_labels(records, path)
    (tmp_path / 'manifest.json').write_text(json.dumps(
        {'sha256': {split: rt._file_sha256(path)}, 'segmenter': {}}))
    return path


def test_segmenter_fingerprint_ignores_line_endings():
    """git checks the source out as CRLF here and LF on Linux; the same code must
    not look like a different segmenter on the other platform."""
    assert rt._source_sha256(b"a = 1\r\nb = 2\r\n") == rt._source_sha256(b"a = 1\nb = 2\n")
    assert rt._source_sha256(b"a = 1\n") != rt._source_sha256(b"a = 2\n")


def test_label_files_are_byte_reproducible(tmp_path):
    """Rebuilding unchanged labels must give an identical file -- gzip embeds a
    timestamp by default, which would make every rebuild look like a change."""
    first, second = tmp_path / 'a.jsonl.gz', tmp_path / 'b.jsonl.gz'

    rt.write_labels(RECORDS, first)
    rt.write_labels(RECORDS, second)

    assert first.read_bytes() == second.read_bytes()


def test_labels_round_trip(tmp_path):
    _store(tmp_path)

    assert rt.load_labels('test', data_dir=tmp_path, check_segmenter=False) == RECORDS


def test_missing_labels_say_how_to_build_them(tmp_path):
    """Labels are generated, not committed, so a fresh clone hits this first."""
    with pytest.raises(FileNotFoundError, match="ragtruth build --split test"):
        rt.load_labels('test', data_dir=tmp_path, check_segmenter=False)


def test_a_manifest_without_the_split_says_how_to_build_it(tmp_path):
    _store(tmp_path, split='test')
    rt.write_labels(RECORDS, rt.labels_path('train', tmp_path))

    with pytest.raises(FileNotFoundError, match="no entry for split 'train'"):
        rt.load_labels('train', data_dir=tmp_path, check_segmenter=False)


def test_a_label_file_that_does_not_match_the_manifest_is_refused(tmp_path):
    path = _store(tmp_path)
    rt.write_labels(RECORDS[:1], path)          # changed after the manifest

    with pytest.raises(ValueError, match="does not match the hash"):
        rt.load_labels('test', data_dir=tmp_path, check_segmenter=False)


# ----------------------------------------------------------------- join

def _labelled(text, spans, sentences):
    return rt.label_response(row(text, spans, query='q', context='c'),
                             segment=lambda _: sentences)


def test_join_restores_sentence_text_from_offsets():
    text = "Water boils at 50 degrees. Ice is cold."
    sentences = spans_of(text, "Water boils at 50 degrees.", "Ice is cold.")
    raw = row(text, [span(text, "50 degrees")], query='q', context='ctx')

    (example,) = rt.join([_labelled(text, [span(text, "50 degrees")], sentences)], [raw])

    assert [s.text for s in example.sentences] == [
        "Water boils at 50 degrees.", "Ice is cold."]
    assert [s.unsupported for s in example.sentences] == [True, False]
    assert example.context == 'ctx'


def test_join_refuses_a_raw_row_whose_output_changed():
    text = "Water boils at 100 degrees. Ice is cold."
    sentences = spans_of(text, "Water boils at 100 degrees.", "Ice is cold.")
    labels = [_labelled(text, [], sentences)]
    edited = row("Water boils at 90 degrees. Ice is cold.", [], query='q', context='c')

    with pytest.raises(ValueError, match="no longer matches"):
        rt.join(labels, [edited])


def test_join_refuses_a_label_with_no_raw_row():
    text = "Water boils at 100 degrees."
    labels = [_labelled(text, [], spans_of(text, text))]

    with pytest.raises(KeyError, match="absent from the raw split"):
        rt.join(labels, [])


def test_unknown_task_type_is_rejected_before_anything_loads():
    with pytest.raises(ValueError, match="unknown task types"):
        rt.load_examples('test', task_types=['Poetry'])


def test_unknown_split_is_rejected_before_any_download():
    with pytest.raises(ValueError, match="unknown split"):
        rt.raw_path('validation')


# ----------------------------------------------- the segmentation contract

def test_sentence_spans_and_sentences_segment_identically():
    """Labels are built from offsets; the auditor scores `split_into_sentences`.
    If these ever disagree, every label describes the wrong sentence."""
    from verify.segmenter import split_into_sentence_spans, split_into_sentences

    text = ("  The grill must be hot. Ok.\n\n1. Season the steak.\n2. Cook it for "
            "4-5 minutes per side!  Rest it, then serve.  ")

    spans = split_into_sentence_spans(text)

    assert [s for _, _, s in spans] == split_into_sentences(text)
    assert all(text[a:b] == s for a, b, s in spans)
    assert all(s == s.strip() and len(s) > 5 for _, _, s in spans)
