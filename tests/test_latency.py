"""E6 latency benchmark: the arithmetic and the report, on synthetic timings.

Nothing here times anything real - that would make the suite slow and flaky.
What is pinned is what the numbers mean: which steps add up to time-to-first-
answer, that the repeated audit is kept out of it, how pacing is computed, and
that a CPU run really hides the GPU.
"""
import json

import pytest

from eval import latency


def test_summarize_reports_milliseconds_and_percentiles():
    stats = latency.summarize([0.010, 0.020, 0.030, 0.040, 1.000])

    assert stats['n'] == 5
    assert stats['p50'] == pytest.approx(30.0)
    assert stats['max'] == pytest.approx(1000.0)
    assert stats['p95'] == pytest.approx(808.0)      # linear interpolation, 40 + 0.8 * 960


def test_summarize_of_nothing_is_nan_not_zero():
    stats = latency.summarize([])

    assert stats['n'] == 0
    assert stats['p50'] != stats['p50']


@pytest.mark.parametrize('last, now, expected', [
    (None, 5.0, 0.0),        # first call never waits
    (0.0, 3.0, 6.0),         # 3 s since the last start, 9 s interval
    (0.0, 12.0, 0.0),        # already late: no negative sleep
])
def test_pace_wait_spaces_call_starts(last, now, expected):
    assert latency.pace_wait(last, now, 9.0) == pytest.approx(expected)


def test_cpu_children_cannot_see_a_gpu(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)

    assert latency.child_env('cpu')['CUDA_VISIBLE_DEVICES'] == '-1'
    assert 'CUDA_VISIBLE_DEVICES' not in latency.child_env('auto')
    with pytest.raises(ValueError, match="'auto' or 'cpu'"):
        latency.child_env('gpu')


# ------------------------------------------------------------------ report

def _cold(scale=1.0, generate=True):
    steps = {'import_streamlit': 0.5, 'import_ingest_index': 1.0, 'import_rag': 0.1,
             'import_verify': 0.4, 'open_index': 2.0, 'load_documents': 0.25,
             'noop_sync': 0.01, 'generator_init': 0.01, 'auditor_init': 3.0,
             'first_retrieve': 0.02, 'first_audit': 0.6, 'second_audit': 0.3}
    if generate:
        steps['first_generate'] = 1.11
    return {'steps': {k: v * scale for k, v in steps.items()}, 'retries': {},
            'process_wall_seconds': 12.0 * scale}


ENV = {'python': '3.10', 'os': 'test-os', 'cpu': 'test-cpu', 'cpu_count': 4, 'torch': '2.x',
       'cuda_available': False, 'gpu': None, 'auditor_device': 'cpu'}

META = {'device_requested': 'cpu', 'reps': 1, 'pace_seconds': 9.0, 'created': 'now',
        'git': {'commit': 'abcdef1234', 'dirty': False},
        'config': {'models': {'generator': 'gen', 'nli_model': 'nli'},
                   'retrieval': {'top_k': 3}, 'verification': {'batch_size': 32}}}

BUILD = {'steps': {'open_empty_index': 1.0, 'load_documents': 0.3,
                   'build_from_empty': 14.0, 'noop_sync': 0.012},
         'chunks': 666, 'added': ['a.pdf', 'b.pdf']}


def _warm(live=True, limited=0):
    audits = [{'id': f'q{i}', 'rep': 0, 'seconds': 0.02 + 0.003 * pairs,
               'segment_seconds': 0.01, 'sentences': pairs // 3, 'pairs': pairs,
               'answer_chars': 100} for i, pairs in enumerate([3, 6, 9, 12, 15, 21])]
    samples = {'retrieve': [{'id': 'q', 'rep': 0, 'seconds': 0.012}] * 6, 'audit': audits,
               'idle': [{'pause': 0.0, 'seconds': 0.065}] * 8 + [{'pause': 9.0, 'seconds': 0.39}] * 8,
               'layers': []}
    if live:
        samples['layers'] = [{'id': 'q', 'plain_first': True, 'plain_generate': 0.80,
                              'plain_prompt_tokens': 90, 'plain_completion_tokens': 300,
                              'retrieve': 0.01, 'rag_generate': 0.95, 'rag_prompt_tokens': 700,
                              'rag_completion_tokens': 250, 'audit': 0.24, 'sentences': 2}] * 4
    return {'samples': samples, 'paced_seconds': 27.0,
            'retries': {'rate_limited': limited, 'connection_errors': 1},
            'environment': ENV}


def test_time_to_first_answer_sums_the_cold_steps_but_not_the_repeat_audit():
    text = latency.build_report(META, BUILD, [_cold()], _warm())

    # 0.5+1+0.1+0.4+2+0.25+0.01+0.01+3+0.02+1.11+0.6 = 9.00 s; second_audit (0.3) excluded
    line = next(l for l in text.splitlines() if 'time to first verified answer' in l)
    assert '9,000 ms' in line


def test_report_recovers_the_per_pair_audit_cost():
    """Audits are built as 20 ms + 3 ms per pair, 10 ms of it segmentation."""
    text = latency.build_report(META, BUILD, [_cold()], _warm())

    assert '~3.0 ms per pair + 10.0 ms fixed' in text


def test_report_separates_what_rag_and_the_checker_each_add():
    """plain 0.80 s; RAG 0.01 + 0.95 = 0.96 s; checked 0.96 + 0.24 = 1.20 s."""
    text = latency.build_report(META, BUILD, [_cold()], _warm(limited=2))

    def row(label):
        return next(l for l in text.splitlines() if label in l)

    assert '800 ms' in row('1. plain Groq, question only')
    assert '960 ms' in row('2. RAG: retrieve + Groq with chunks')
    assert '1,200 ms' in row('3. RAG + RagShield checker (the app)')
    assert '160 ms' in row('RAG over plain Groq')
    assert '240 ms' in row('checker over RAG')
    assert "checker share of the app's total: median 20% per question, 20% of all time" in text
    assert 'prompt 90 vs 700, completion (includes reasoning) 300 vs 250' in text
    assert '2 rate limits (429), 1 connection errors' in text
    assert 'pacing added 27 s' in text
    assert 'build from empty (666 chunks, 2 PDFs)' in text


def test_cold_start_names_the_checkers_share():
    text = latency.build_report(META, BUILD, [_cold()], _warm())

    # import_verify 0.4 + auditor_init 3.0 + first_audit 0.6
    line = next(l for l in text.splitlines() if 'of which checker' in l)
    assert '4,000 ms' in line


def test_report_compares_back_to_back_with_idle_audits():
    text = latency.build_report(META, BUILD, [_cold()], _warm())

    assert 'audit, back-to-back' in text and '65.0 ms' in text
    assert 'audit after 9 s idle' in text and '390 ms' in text


def test_report_of_a_run_without_idle_samples_still_renders():
    warm = _warm()
    del warm['samples']['idle']

    assert 'IDLE' not in latency.build_report(META, BUILD, [_cold()], warm)


def test_report_without_live_queries_says_so():
    text = latency.build_report(META, BUILD, [_cold(generate=False)], _warm(live=False))

    assert 'WHAT EACH LAYER ADDS' not in text
    assert 'no live generate in cold runs' in text


def test_report_reads_a_saved_run(tmp_path):
    for name, payload in [('run.json', META), ('build.json', BUILD), ('warm.json', _warm()),
                          ('cold0.json', _cold()), ('cold1.json', _cold(scale=2.0))]:
        (tmp_path / name).write_text(json.dumps(payload), encoding='utf-8')

    text = latency.report(tmp_path)

    assert 'COLD START - fresh process, 2 runs' in text
    assert (tmp_path / 'report.txt').read_text(encoding='utf-8') == text
