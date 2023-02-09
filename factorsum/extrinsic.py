import logging

import nltk
from summa import summarizer
import numpy as np


logger = logging.getLogger(__name__)
try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)


def _candidate_score(candidate_summary, guidance):
    candidate_score = 0

    if guidance:
        if type(guidance) != list:
            guidance = [guidance]

        for guidance_item in guidance:
            candidate_score += guidance_item.score(candidate_summary)

    return candidate_score


def _add_best_view_fast(summary, views, guidance, constraints, current_score, deltas):
    """
    An optimized implementation that takes submodularity of the function into account.

    Key idea is to sort summary views by deltas (discrete derivatives) and check if
    the new delta is larger than other previous deltas. For details see page 9 here:
    https://viterbi-web.usc.edu/~shanghua/teaching/Fall2021-670/krause12survey.pdf
    """

    best_score = None
    best_summary = None
    if deltas is None:
        deltas = [np.inf] * len(views)

    if current_score is None:
        current_score = 0

    view_idxs = np.argsort(deltas)[::-1]
    for view_idx in view_idxs:
        view = views[view_idx]

        if view in summary:
            deltas[view_idx] = 0
            continue

        if not all([c.check(summary, view) for c in constraints]):
            deltas[view_idx] = 0
            continue

        candidate_summary = summary + [view]
        score = _candidate_score(candidate_summary, guidance)
        # we normalize by the "cost" of the view to optimize a submodular
        # function with a knapsack constraint. See Eq. 13 here:
        # https://viterbi-web.usc.edu/~shanghua/teaching/Fall2021-670/krause12survey.pdf
        delta = score - current_score
        view_cost = len(nltk.word_tokenize(view))
        delta = delta / (1.0 * view_cost)
        deltas[view_idx] = delta

        if np.argmax(deltas) == view_idx:
            best_summary = candidate_summary
            best_score = score
            deltas[view_idx] = 0
            break

    if sum(deltas) == 0:
        best_summary = summary
        best_score = None

    elif best_score is None:
        max_idx = np.argmax(deltas)
        best_view = views[max_idx]
        best_summary = summary + [best_view]
        view_cost = len(nltk.word_tokenize(best_view))
        best_score = deltas[max_idx] * view_cost + current_score
        deltas[max_idx] = 0

    return best_summary, best_score, deltas


def _check_constraint(constraint, summary, view):
    return constraint.check(summary, view)


def _add_best_view(summary, views, guidance, constraints, current_score):
    best_score = None
    best_delta = None
    best_summary = None
    if current_score is None:
        current_score = 0

    for view in views:
        if view in summary:
            continue

        if not all([_check_constraint(c, summary, view) for c in constraints]):
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


def greedy_summary(
    views,
    guidance,
    constraints,
    patience=-1,
):
    summary = []
    best_summary = []
    attempts = 0
    best_score = None
    # deltas = None

    while True:
        # _summary, score, deltas = _add_best_view_fast(
        #     summary, views, guidance, constraints, best_score, deltas
        # )
        # logger.info(score, best_score)

        _summary, score = _add_best_view(
            summary, views, guidance, constraints, best_score
        )
        # logger.info(best_score, score)

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


def textrank_summary(views, token_budget):
    views = "\n".join(views)
    if len(nltk.word_tokenize(views)) < token_budget:
        summary = views
    else:
        summary = summarizer.summarize(views, words=token_budget)

    summary = summary.split("\n")
    return summary
