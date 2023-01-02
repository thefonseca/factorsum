import nltk
from summa import summarizer

from .constraints import SummaryViewConstraint, BudgetConstraint, RedundancyConstraint
from .guidance import ROUGEContentGuidance
from .utils import apply_word_limit, show_summary, sent_tokenize_views
from .oracle import get_oracles

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)


def _get_valid_views(views, min_words=5):
    n_ignored = sum([p is None for p in views])
    if n_ignored > 0:
        print(f"Ignoring {n_ignored} predicted summary")
        views = [p for p in views if p is not None]

    _views = []

    constraint = SummaryViewConstraint(min_length=min_words)

    for view in views:
        if not constraint.check(view):
            continue

        if view not in _views:
            _views.append(view)

    return _views


def _preprocess_summary_views(views, min_words_per_view):
    views = _get_valid_views(views, min_words=min_words_per_view)
    # sent tokenize all preds to get more fine-grained information
    views = sent_tokenize_views(views, min_words=min_words_per_view)
    return views


def _candidate_score(candidate_summary, guidance):
    candidate_score = 0

    if guidance:
        if type(guidance) != list:
            guidance = [guidance]

        for guidance_item in guidance:
            candidate_score += guidance_item.score(candidate_summary)

    return candidate_score


def _get_oracle_idxs(source_sents, target_sents):
    oracle_idxs = get_oracles([source_sents], [target_sents], progress_bar=False)[0]
    oracle_idxs = [x if x is not None else len(oracle_idxs) for x in oracle_idxs]
    return oracle_idxs


def _reorder_sentences(summary, target_content):
    _target_content = target_content.replace("<n>", "")
    _target_content = nltk.sent_tokenize(_target_content)
    oracle_idxs = _get_oracle_idxs(_target_content, summary)
    summary = [x for _, x in sorted(zip(oracle_idxs, summary))]
    return summary


def _add_best_view(summary, views, guidance, constraints, current_score):
    best_score = None
    best_delta = None
    best_summary = None
    if current_score is None:
        current_score = 0

    for view in views:
        if view in summary:
            continue

        if not all([c.check(summary, view) for c in constraints]):
            continue

        candidate_summary = summary + [view]
        score = _candidate_score(candidate_summary, guidance)
        # we normalize by the "cost" of the view to optimize a submodular
        # function with a knapsack constraint. See Eq. 13 here:
        # https://viterbi-web.usc.edu/~shanghua/teaching/Fall2021-670/krause12survey.pdf
        delta = score - current_score
        view_cost = len(nltk.word_tokenize(view))
        delta = delta / (1.0 * view_cost)

        if best_delta is None or best_delta < delta:
            best_summary = candidate_summary
            best_score = score
            best_delta = delta

    return best_summary, best_score


def _greedy_summary(
    views,
    guidance,
    constraints,
    patience=-1,
):
    summary = []
    best_summary = []
    attempts = 0
    best_score = None

    while True:
        _summary, score = _add_best_view(
            summary, views, guidance, constraints, best_score
        )

        if _summary is None or len(summary) == len(_summary):
            break
        if best_score is not None and best_score > score:
            attempts += 1
            if patience >= 0 and attempts > patience:
                break
        else:
            best_score = score
            best_summary = _summary
            attempts = 0

        summary = _summary

    return best_summary


def _textrank_summary(views, token_budget):
    views = "\n".join(views)
    if len(nltk.word_tokenize(views)) < token_budget:
        summary = views
    else:
        summary = summarizer.summarize(views, words=token_budget)

    summary = summary.split("\n")
    return summary


def _get_guidance(
    target_content,
    content_weight=1.0,
    custom_guidance=None,
):
    guidance = []
    if target_content:
        guidance.append(ROUGEContentGuidance(target_content, weight=content_weight))

    if custom_guidance:
        if type(custom_guidance) == list:
            guidance.extend(custom_guidance)
        else:
            guidance.append(custom_guidance)

    return guidance


def _get_constraints(target_budget, custom_constraints=None):
    constraints = [BudgetConstraint(target_budget), RedundancyConstraint()]
    if custom_constraints:
        if type(custom_constraints) == list:
            constraints.extend(custom_constraints)
        else:
            constraints.append(custom_constraints)
    return constraints


def find_best_summary(
    summary_views,
    target_budget,
    strict_budget=False,
    target_content=None,
    content_weight=1.0,
    custom_guidance=None,
    custom_constraints=None,
    verbose=False,
    method="factorsum",
    min_words_per_view=5,
):
    summary = []

    if target_content is not None:
        if type(target_content) == list:
            target_content = "\n".join(target_content)

    guidance = _get_guidance(
        target_content,
        content_weight=content_weight,
        custom_guidance=custom_guidance,
    )
    constraints = _get_constraints(target_budget, custom_constraints=custom_constraints)
    summary_views = _preprocess_summary_views(summary_views, min_words_per_view)

    if method == "factorsum":
        summary = _greedy_summary(summary_views, guidance, constraints)
    elif method == "textrank":
        summary = _textrank_summary(summary_views, target_budget)
    else:
        raise ValueError(f"Unsupported summarization method: {method}")

    if strict_budget:
        summary = apply_word_limit(summary, target_budget, return_list=True)

    if target_content:
        summary = _reorder_sentences(summary, target_content)

    guidance_scores = {}
    if guidance:
        guidance_scores = {g.__class__.__name__: g.score(summary) for g in guidance}

    if verbose:
        print()
        show_summary(summary)

    return summary, guidance_scores
