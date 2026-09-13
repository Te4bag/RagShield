"""The E3 runner, against a stub auditor. No weights, no dataset download.

What is pinned here is the part that decides what the numbers *mean*: the
sentence -> answer rule fixed before any run, the answer gold coming from
RAGTruth's spans rather than from segmentation, alignment between scored and
labelled sentences, truncation accounting, and the caveats the report must
carry next to its figures.
"""
import numpy as np
import pytest

from eval import run
from eval.datasets.ragtruth import Example, Sentence


def sentence(text, unsupported=False, label_types=(), implicit_true=False):
    return Sentence(text, 0, len(text), unsupported, tuple(label_types),
                    implicit_true, False)


def example(eid, sentences, context="Passage one.\n\nPassage two.", n_spans=None,
            task_type='QA'):
    if n_spans is None:
        n_spans = sum(s.unsupported for s in sentences)
    return Example(id=eid, task_type=task_type, model='m', quality='good', query='q',
                   context=context, output=' '.join(s.text for s in sentences),
                   sentences=tuple(sentences), n_spans=n_spans)


class StubTokenizer:
    """Token count = words, so a test can aim a pair over the budget."""
    def __call__(self, premises, hypotheses, truncation=None):
        return {'input_ids': [[0] * (len(p.split()) + len(h.split()))
                              for p, h in zip(premises, hypotheses)]}


class StubAuditor:
    """Speaks the NLIAuditor interface the runner uses, with scripted P(entailment)."""
    def __init__(self, p_entailment, max_length=512, sentences=None):
        self.p_entailment = p_entailment          # sentence text -> P(entailment)
        self.aggregation = 'max_entailment'
        self.threshold = 0.85
        self.max_length = max_length
        self.model = type('M', (), {'tokenizer': StubTokenizer()})()
        self.sentences = sentences
        self.calls = []

    def audit_response(self, text, premises):
        self.calls.append((self.aggregation, [p['text'] for p in premises]))
        names = self.sentences if self.sentences is not None else self._split(text)
        out = []
        for s in names:
            p = self.p_entailment[s]
            rest = (1.0 - p) / 2
            out.append({'sentence': s,
                        'verdict': 'ENTAILMENT' if p >= self.threshold else 'NEUTRAL',
                        'entailment': p,
                        'probabilities': {'ENTAILMENT': p, 'NEUTRAL': rest,
                                          'CONTRADICTION': rest}})
        return out

    @staticmethod
    def _split(text):
        return [t if t.endswith('.') else t + '.' for t in
                (part.strip() for part in text.split('. ')) if t]


# ------------------------------------------------------------------ context

def test_short_passages_stay_separate_premises():
    """QA contexts are three retrieved passages; merging them would score a
    sentence against two passages at once, which the app never does."""
    context = "\n\n".join(["alpha " * 40, "beta " * 40, "gamma " * 40])

    premises = run.context_premises(context)

    assert len(premises) == 3
    assert [p['text'].split()[0] for p in premises] == ['alpha', 'beta', 'gamma']


def test_a_long_context_is_cut_at_the_app_chunk_size():
    from index import cfg

    premises = run.context_premises("word " * 1000)

    assert len(premises) > 1
    assert all(len(p['text']) <= cfg['ingestion']['chunk_size'] for p in premises)


# ------------------------------------------------------------------ scoring

def test_score_example_joins_probabilities_to_labels():
    ex = example('1', [sentence("Paris is in France."),
                       sentence("It has ten moons.", unsupported=True,
                                label_types=['Evident Conflict'])])
    auditor = StubAuditor({"Paris is in France.": 0.97, "It has ten moons.": 0.02})

    rec = run.score_example(ex, auditor)

    assert [s['p_entailment'] for s in rec['sentences']] == [0.97, 0.02]
    assert [s['unsupported'] for s in rec['sentences']] == [False, True]
    assert rec['sentences'][1]['label_types'] == ['Evident Conflict']
    assert rec['n_premises'] == 2
    assert rec['n_spans'] == 1


def test_scored_sentences_must_match_labelled_ones():
    """If the auditor segments differently from the labels, every label would
    describe the wrong sentence. That must stop the run, not skew it."""
    ex = example('1', [sentence("One sentence."), sentence("Two sentence.")])
    auditor = StubAuditor({"One sentence.": 0.9}, sentences=["One sentence."])

    with pytest.raises(run.AlignmentError, match="response 1"):
        run.score_example(ex, auditor)


def test_truncated_pairs_are_counted_per_sentence():
    long_context = "\n\n".join(["short passage here", "x " * 30])
    ex = example('1', [sentence("Tiny claim.")], context=long_context)
    auditor = StubAuditor({"Tiny claim.": 0.5}, max_length=20)

    rec = run.score_example(ex, auditor)

    # 3+2 words fits in 20; 30+2 does not.
    assert rec['sentences'][0]['max_pair_tokens'] == 32
    assert rec['sentences'][0]['truncated_pairs'] == 1


def test_concatenate_counts_tokens_against_the_joined_premise():
    context = "\n\n".join(["a " * 12, "b " * 12])
    ex = example('1', [sentence("Tiny claim.")], context=context)
    auditor = StubAuditor({"Tiny claim.": 0.5}, max_length=20)
    auditor.aggregation = 'concatenate'

    rec = run.score_example(ex, auditor)

    assert rec['n_premises'] == 1
    assert rec['sentences'][0]['max_pair_tokens'] == 24 + 2
    assert rec['sentences'][0]['truncated_pairs'] == 1


def test_score_examples_sets_the_aggregation_on_the_auditor():
    ex = example('1', [sentence("Claim here.")])
    auditor = StubAuditor({"Claim here.": 0.5})

    run.score_examples([ex], auditor, 'concatenate', progress=False)

    assert auditor.calls[0][0] == 'concatenate'


def test_unknown_aggregation_is_rejected_before_scoring():
    auditor = StubAuditor({})

    with pytest.raises(ValueError, match="unknown aggregation"):
        run.score_examples([], auditor, 'majority_vote', progress=False)
    assert auditor.calls == []


def test_a_sentence_with_no_premise_scores_zero_entailment():
    ex = example('1', [sentence("Claim here.")], context="")

    class NoPremise(StubAuditor):
        def audit_response(self, text, premises):
            return [{'sentence': 'Claim here.', 'verdict': 'NEUTRAL', 'entailment': None,
                     'probabilities': None, 'evidence': None}]

    rec = run.score_example(ex, NoPremise({}))

    assert rec['sentences'][0]['p_entailment'] == 0.0
    assert rec['n_premises'] == 0


# ---------------------------------------------------- sentence -> answer rule

def rec(eid, p_entailment, n_spans, unsupported=None, task='QA', implicit=None):
    unsupported = unsupported or [False] * len(p_entailment)
    implicit = implicit or [False] * len(p_entailment)
    return {'id': eid, 'task_type': task, 'model': 'm', 'quality': 'good',
            'n_spans': n_spans, 'n_premises': 3,
            'sentences': [{'unsupported': u, 'label_types': ['Evident Conflict'] if u else [],
                           'implicit_true': i, 'due_to_null': False,
                           'verdict': 'ENTAILMENT' if p >= 0.85 else 'NEUTRAL',
                           'p_entailment': p, 'p_contradiction': 0.0, 'p_neutral': 1 - p,
                           'max_pair_tokens': 100, 'truncated_pairs': 0}
                          for p, u, i in zip(p_entailment, unsupported, implicit)]}


def test_one_sentence_below_tau_flags_the_whole_answer():
    gold, score, predicted, empty = run.answer_level(
        [rec('a', [0.99, 0.95, 0.40], n_spans=1),
         rec('b', [0.99, 0.90], n_spans=0)], tau=0.85)

    assert predicted.tolist() == [True, False]
    assert score == pytest.approx([0.60, 0.10])
    assert gold.tolist() == [True, False]
    assert empty == 0


def test_the_rule_is_strict_at_tau():
    """P(entailment) exactly at tau is green in the app, so not flagged here."""
    _, _, predicted, _ = run.answer_level([rec('a', [0.85], n_spans=0)], tau=0.85)

    assert predicted.tolist() == [False]


def test_answer_gold_comes_from_spans_not_from_sentence_labels():
    """RAGTruth's answer label is 'has any span'; it must not depend on
    whether a span happened to land on a kept sentence."""
    gold, _, _, _ = run.answer_level([rec('a', [0.99], n_spans=2, unsupported=[False])],
                                     tau=0.85)

    assert gold.tolist() == [True]


def test_an_answer_with_no_scorable_sentence_is_predicted_clean_and_counted():
    gold, score, predicted, empty = run.answer_level([rec('a', [], n_spans=1)], tau=0.85)

    assert predicted.tolist() == [False]
    assert score.tolist() == [0.0]
    assert gold.tolist() == [True]
    assert empty == 1


# ------------------------------------------------------------------ reporting

def _meta(**overrides):
    meta = {'split': 'test', 'n_responses': 4, 'limit': None, 'seed': 0,
            'labels_sha256': 'ab' * 32,
            'dataset': {'repo': 'wandb/RAGTruth-processed', 'revision': 'eb4f4b9d' + '0' * 32},
            'git': {'commit': '1234567abcdef', 'dirty': False},
            'environment': {'device': 'cpu', 'gpu': None},
            'config': {'nli_model': 'stub', 'verification': {'entailment_threshold': 0.85}},
            'aggregations': ['max_entailment']}
    meta.update(overrides)
    return meta


RECORDS = [
    rec('1', [0.99, 0.20], n_spans=1, unsupported=[False, True]),
    rec('2', [0.95, 0.97], n_spans=0),
    rec('3', [0.30, 0.90], n_spans=1, unsupported=[True, False]),
    rec('4', [0.60], n_spans=0),
]


def test_report_states_the_rule_tau_hash_and_comparability_next_to_the_numbers():
    text = run.build_report(_meta(), {'max_entailment': RECORDS}, tau=0.85)

    assert run.ANSWER_RULE in text
    assert 'tau = 0.85' in text
    assert 'ab' * 32 in text
    assert 'only QA is directly comparable' in text
    assert 'HEADLINE - QA, max_entailment' in text


def test_headline_answer_f1_follows_the_rule():
    """By hand: predicted hallucinated = answers 1, 3, 4 (min P < .85); gold = 1, 3.
    TP 2, FP 1, FN 0 -> P 2/3, R 1, F1 0.8."""
    s = run.scope_summary(RECORDS, tau=0.85)
    positive = s['answer_verdict'].per_class[0]

    assert positive.label == 'hallucinated'
    assert positive.precision == pytest.approx(2 / 3)
    assert positive.recall == pytest.approx(1.0)
    assert positive.f1 == pytest.approx(0.8)
    assert 'answer-level F1 80.0' in run.build_report(_meta(), {'max_entailment': RECORDS}, 0.85)


def test_flag_all_f1_is_the_base_rate_reference():
    """Half the answers hallucinated: flagging all gives P .5, R 1, F1 2/3."""
    assert run.flag_all_f1([True, False, True, False]) == pytest.approx(2 / 3)
    assert run.flag_all_f1([False, False]) == 0.0


def test_a_rule_that_flags_everything_scores_exactly_the_flag_all_f1():
    """The trap the reference column exists for: at recall 1 and precision =
    base rate, the rule's F1 *is* the baseline, however good it looks."""
    records = [rec('1', [0.1], n_spans=1), rec('2', [0.2], n_spans=0),
               rec('3', [0.3], n_spans=0), rec('4', [0.4], n_spans=1)]

    s = run.scope_summary(records, tau=0.85)

    assert s['answer_verdict'].per_class[0].f1 == pytest.approx(s['answer_flag_all_f1'])


def test_oracle_f1_finds_the_best_threshold_on_the_same_labels():
    """Scores 1 - min P for RECORDS: .80 (gold), .05, .70 (gold), .40.
    Flagging the top two catches both golds with no false positive -> F1 1.0,
    at cutoff min P(entailment) <= 1 - .70 = .30."""
    s = run.scope_summary(RECORDS, tau=0.85)

    assert s['answer_oracle_f1'] == pytest.approx(1.0)
    assert s['answer_oracle_cutoff'] == pytest.approx(0.30)


def test_oracle_f1_is_undefined_on_a_single_class():
    f1, cutoff = run.oracle_f1([True, True], [0.2, 0.8])

    assert np.isnan(f1) and np.isnan(cutoff)


def test_report_labels_the_oracle_as_a_ceiling_not_a_result():
    text = run.build_report(_meta(), {'max_entailment': RECORDS}, tau=0.85)

    assert 'flag-every-answer baseline F1 66.7' in text
    assert 'picked ON TEST' in text and 'not a result' in text


def test_changing_tau_changes_the_answer_prediction_not_the_auroc():
    low = run.scope_summary(RECORDS, tau=0.5)
    high = run.scope_summary(RECORDS, tau=0.85)

    assert low['answer_verdict'].per_class[0].predicted == 2
    assert high['answer_verdict'].per_class[0].predicted == 3
    assert low['answer_detection'].pooled_auroc == high['answer_detection'].pooled_auroc


def test_an_overridden_tau_is_labelled_as_such():
    text = run.build_report(_meta(), {'max_entailment': RECORDS}, tau=0.6)

    assert "overridden; this run's config has entailment_threshold 0.85" in text


def test_tau_defaults_to_the_derived_floor_when_the_run_recorded_one():
    meta = _meta(config={'nli_model': 'stub', 'verification': {
        'entailment_threshold': 0.85, 'low_support_threshold': 0.000552}})

    assert run.config_tau(meta) == (0.000552, 'low_support_threshold')
    assert 'derived on train in E4' in run.build_report(meta, {'max_entailment': RECORDS},
                                                        tau=0.000552)


def test_a_run_scored_before_e4_falls_back_to_its_own_gate():
    """Older run.json files have no floor; their reports must not change."""
    assert run.config_tau(_meta()) == (0.85, 'entailment_threshold')


def test_a_single_class_scope_reports_without_crashing():
    clean_only = [rec('1', [0.99], n_spans=0), rec('2', [0.95], n_spans=0)]

    text = run.build_report(_meta(), {'max_entailment': clean_only}, tau=0.85)

    assert 'ABLATION' in text


def test_ablation_table_has_a_row_per_aggregation_and_scope():
    records = RECORDS + [rec('5', [0.4, 0.99], n_spans=1, unsupported=[True, False],
                             task='Summary'),
                         rec('6', [0.99, 0.99], n_spans=0, task='Summary')]

    text = run.build_report(_meta(aggregations=['max_entailment', 'concatenate']),
                            {'max_entailment': records, 'concatenate': records}, 0.85)

    table = text.split('ABLATION')[1].split('within =')[0]
    for aggregation in ('max_entailment', 'concatenate'):
        for scope in ('QA', 'Summary', 'ALL'):
            assert any(line.startswith(aggregation) and f' {scope} ' in f' {line} '
                       for line in table.splitlines())


def test_crosstab_shows_only_the_verdicts_a_run_produced():
    """Pre-P8 runs have CONTRADICTION and no LOW_SUPPORT; later runs the reverse."""
    old = [{'unsupported': False, 'label_types': [], 'verdict': v}
           for v in ('ENTAILMENT', 'CONTRADICTION')]
    new = [{'unsupported': True, 'label_types': ['Evident Conflict'], 'verdict': v}
           for v in ('NEUTRAL', 'LOW_SUPPORT')]

    old_header = run._crosstab(old).splitlines()[0]
    new_header = run._crosstab(new).splitlines()[0]

    assert 'CONTRADICTION' in old_header and 'LOW_SUPPORT' not in old_header
    assert 'LOW_SUPPORT' in new_header and 'CONTRADICTION' not in new_header


def test_implicit_true_sensitivity_excludes_those_sentences():
    records = RECORDS + [rec('5', [0.1, 0.99], n_spans=1, unsupported=[True, False],
                             implicit=[True, False])]

    s = run.scope_summary(records, tau=0.85)

    assert s['implicit_true_sentences'] == 1
    assert s['without_implicit_true'].n == s['n_sentences'] - 1


def test_sentence_arrays_keep_response_ids_for_within_response_auroc():
    gold, p_ent, groups, _ = run.sentence_level(RECORDS)

    assert groups == ['1', '1', '2', '2', '3', '3', '4']
    assert gold.tolist() == [False, True, False, False, True, False, False]
    assert p_ent == pytest.approx(np.array([0.99, 0.20, 0.95, 0.97, 0.30, 0.90, 0.60]))
