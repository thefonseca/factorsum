import nltk
from summa import summarizer

from .score import extrinsic_scores
from .utils import apply_word_limit, show_summary, sent_tokenize_views
from .oracle import get_oracles
from .redundancy import has_similar_content, has_repeated_trigram

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)


def _get_valid_view_idxs(views, min_words=5):
    n_ignored = sum([p is None for p in views])
    if n_ignored > 0:
        print(f"Ignoring {n_ignored} predicted summary")
        views = [p for p in views if p is not None]

    view_idxs = []
    _views = []

    for idx, view in enumerate(views):
        has_min_length = len(view.split()) > min_words if min_words else True

        if not has_min_length:
            continue

        if view not in _views:
            _views.append(view)
            view_idxs.append(idx)

    return view_idxs, _views


def _candidate_score(
    candidate_summary,
    content_guidance=None,
    budget_guidance=None,
    content_weight=1.0,
    budget_weight=1.0,
):

    candidate_score = 0

    scores = extrinsic_scores(
        candidate_summary,
        target_summary=content_guidance,
        token_budget=budget_guidance,
        rouge_ngrams=["rouge1"],
    )

    if "rouge" in scores:
        candidate_score += content_weight * scores["rouge"]["rouge1"].fmeasure
    if "budget_error" in scores:
        candidate_score += -budget_weight * scores["budget_error"]

    return candidate_score


def _add_best_view(
    summary,
    summary_idxs,
    views,
    view_idxs,
    content_guidance=None,
    budget_guidance=None,
    content_weight=1.0,
    budget_weight=1.0,
    trigrams=None,
):

    best_score = None
    best_summary = None
    best_idxs = None
    best_trigrams = list(trigrams) if trigrams is not None else None

    for ii in view_idxs:
        if ii in summary_idxs:
            continue

        candidate_trigrams = None
        if trigrams is not None:
            candidate_trigrams = list(trigrams)
            if has_repeated_trigram(views[ii], candidate_trigrams):
                continue
        else:
            if has_similar_content(views[ii], summary):
                continue

        candidate_summary = summary + [views[ii]]
        idxs = summary_idxs + [ii]

        score = _candidate_score(
            candidate_summary,
            content_guidance=content_guidance,
            budget_guidance=budget_guidance,
            content_weight=content_weight,
            budget_weight=budget_weight,
        )

        if best_score is None or best_score < score:
            best_summary = candidate_summary
            best_score = score
            best_idxs = idxs
            best_trigrams = candidate_trigrams

    return best_summary, best_idxs, best_score, best_trigrams


def _get_oracle_idxs(source_sents, target_sents):
    oracle_idxs = get_oracles([source_sents], [target_sents])[0]
    oracle_idxs = [x if x is not None else len(oracle_idxs) for x in oracle_idxs]
    return oracle_idxs


def _reorder_sentences(summary, summary_idxs, content_guidance):
    _content_guidance = content_guidance.replace("<n>", "")
    _content_guidance = nltk.sent_tokenize(_content_guidance)
    oracle_idxs = _get_oracle_idxs(_content_guidance, summary)
    summary = [x for _, x in sorted(zip(oracle_idxs, summary))]
    summary_idxs = [x for _, x in sorted(zip(oracle_idxs, summary_idxs))]
    return summary, summary_idxs


def _find_best_summary(
    summary,
    summary_idxs,
    views,
    view_idxs,
    content_guidance=None,
    content_weight=1.0,
    budget_weight=1.0,
    budget_guidance=None,
    patience=-1,
):

    best_summary = []
    best_idxs = []
    attempts = 0
    best_score = None

    while True:

        result = _add_best_view(
            summary,
            summary_idxs,
            views,
            view_idxs,
            content_guidance=content_guidance,
            budget_guidance=budget_guidance,
            content_weight=content_weight,
            budget_weight=budget_weight,
            trigrams=None,
        )
        _summary, _idxs, score, _ = result

        if _summary is None or len(summary) == len(_summary):
            break
        if best_score is not None and best_score > score:
            attempts += 1
            if patience >= 0 and attempts > patience:
                break
        else:
            best_score = score
            best_summary = _summary
            best_idxs = _idxs
            attempts = 0

        summary = _summary
        summary_idxs = _idxs

    return best_summary, best_idxs


def _greedy_summary(
    views,
    content_guidance=None,
    budget_guidance=None,
    content_weight=1.0,
    budget_weight=1.0,
    patience=-1,
    min_words=5,
):

    summary = []
    summary_idxs = []

    view_idxs, _ = _get_valid_view_idxs(views, min_words=min_words)

    # sent tokenize all preds to get more fine-grained information
    result = sent_tokenize_views(views, view_idxs)
    sents, sent_idxs, sent_to_view_idxs = result

    # find best greedy summary limited by patience
    result = _find_best_summary(
        summary,
        summary_idxs,
        sents,
        sent_idxs,
        content_guidance=content_guidance,
        budget_guidance=budget_guidance,
        content_weight=content_weight,
        budget_weight=budget_weight,
        patience=patience,
    )
    summary, idxs = result

    # translate back to view-level index
    summary_idxs = [sent_to_view_idxs[idx] for idx in idxs]

    assert len(summary) == len(summary_idxs), f"{len(summary)} != {len(summary_idxs)}"
    return summary, summary_idxs


def _textrank_summary(views, token_budget, min_words=5):

    _, views = _get_valid_view_idxs(views, min_words=min_words)
    views = "\n".join(views)
    if len(nltk.word_tokenize(views)) < token_budget:
        summary = views
    else:
        summary = summarizer.summarize(views, words=token_budget)

    summary = summary.split("\n")
    return summary


def find_best_summary(
    summary_views,
    budget_guidance=200,
    strict_budget=False,
    content_guidance=None,
    content_weight=1.0,
    budget_weight=1.0,
    verbose=False,
    method="factorsum",
):

    summary = []
    summary_idxs = []

    if content_guidance is not None:
        if type(content_guidance) == list:
            content_guidance = "\n".join(content_guidance)

    if method == "factorsum":
        summary, summary_idxs = _greedy_summary(
            summary_views,
            budget_guidance=budget_guidance,
            content_guidance=content_guidance,
            content_weight=content_weight,
            budget_weight=budget_weight,
        )
    elif method == "textrank":
        summary = _textrank_summary(summary_views, budget_guidance)
    else:
        raise ValueError(f"Unsupported summarization method: {method}")

    if strict_budget:
        summary = apply_word_limit(summary, budget_guidance, return_list=True)

    if content_guidance and summary_idxs:
        summary, summary_idxs = _reorder_sentences(
            summary, summary_idxs, content_guidance
        )

    if verbose:
        print()
        show_summary(summary)

    return summary, summary_idxs
