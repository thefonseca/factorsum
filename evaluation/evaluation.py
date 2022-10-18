import pathlib
import json

import numpy as np
import pandas as pd
from rouge_score import scoring
import nltk
from p_tqdm import p_map

from .utils import log_summary, log_rouge_scores
from factorsum.extrinsic import find_best_summary
from factorsum.utils import apply_word_limit
from factorsum.score import extrinsic_scores


def _print_eval_metrics(results):
    print("Number of documents:", len(results["sents_per_summary"]))
    print("Avg sentences per summary:", np.mean(results["sents_per_summary"]))
    print("Avg tokens per summary:", np.mean(results["tokens_per_summary"]))
    print("Avg tokens per summary sent:", np.mean(results["tokens_per_summary_sent"]))
    print("Avg sentences per abstract:", np.mean(results["sents_per_abstract"]))
    print("Avg tokens per abstract:", np.mean(results["tokens_per_abstract"]))
    print("Avg tokens per abstract sent:", np.mean(results["tokens_per_abstract_sent"]))
    print("Avg token difference:", np.mean(results["length_diffs"]))
    print()

    scores = results["scores"]
    print("\n> ROUGE scores")
    log_rouge_scores(scores["rouge"])


def _aggregate_parallel_results(p_results):
    results = {}

    for result in p_results:
        for key in result.keys():
            if type(result[key]) == dict:
                dict_values = results[key]
                for dict_key in result[key]:
                    values = dict_values.get(dict_key, [])
                    values.extend(result[key][dict_key])
                    dict_values[dict_key] = values
            else:
                values = results.get(key, [])
                values.extend(result[key])
                results[key] = values
    return results


def _aggregate_results(results):

    if type(results) == list:
        results = _aggregate_parallel_results(results)

    scores = {}
    aggregator = scoring.BootstrapAggregator()
    for rouge_score in results["rouge_scores"]:
        aggregator.add_scores(rouge_score)
    scores["rouge"] = aggregator.aggregate()

    results["scores"] = scores
    return results


def eval_job(
    pred,
    target,
    doc_id,
    good_score=None,
    bad_score=None,
    max_target_tokens=None,
):

    sents_per_summary = []
    tokens_per_summary = []
    tokens_per_summary_sent = []
    sents_per_abstract = []
    tokens_per_abstract = []
    tokens_per_abstract_sent = []
    length_diffs = []
    rouge_scores = []

    if type(pred) == list:
        pred_sents = pred
    else:
        pred_sents = nltk.sent_tokenize(pred)

    pred = "\n".join(pred_sents)

    try:
        if target is None or str(target) == "nan" or len(target) == 0:
            return
    except:
        print(target)

    target = apply_word_limit(target, max_target_tokens)
    target_is_list = type(target) == list
    if target_is_list:
        target = "\n".join(target)

    pred_words = nltk.word_tokenize(pred)
    target_words = nltk.word_tokenize(target)

    sents_per_summary_doc = len(pred_sents)
    tokens_per_summary_sent_doc = []
    for sent in pred_sents:
        words = nltk.word_tokenize(sent)
        tokens_per_summary_sent_doc.append(len(words))
    if len(tokens_per_summary_sent_doc):
        tokens_per_summary_sent_doc = np.mean(tokens_per_summary_sent_doc)
    else:
        tokens_per_summary_sent_doc = 0
    tokens_per_summary_doc = len(pred_words)
    length_diff = len(pred_words) - len(target_words)

    length_diffs.append(length_diff)
    sents_per_summary.append(sents_per_summary_doc)
    tokens_per_summary_sent.append(tokens_per_summary_sent_doc)
    tokens_per_summary.append(tokens_per_summary_doc)

    if target_is_list:
        target_sents = target.split("\n")
    else:
        target_sents = nltk.sent_tokenize(target)
    sents_per_abstract.append(len(target_sents))
    tokens_per_abstract_sent.append(np.mean([len(s.split()) for s in target_sents]))
    tokens_per_abstract.append(len(target_words))

    score = extrinsic_scores(pred_sents, target_summary=target)
    rouge_score = score["rouge"]

    if rouge_score:
        rouge_scores.append(rouge_score)

    log_summary(doc_id, pred, target, score, bad_score, good_score)

    results = dict(
        sents_per_summary=sents_per_summary,
        tokens_per_summary=tokens_per_summary,
        tokens_per_summary_sent=tokens_per_summary_sent,
        sents_per_abstract=sents_per_abstract,
        tokens_per_abstract=tokens_per_abstract,
        tokens_per_abstract_sent=tokens_per_abstract_sent,
        length_diffs=length_diffs,
        rouge_scores=rouge_scores,
    )
    return results


def evaluate(
    preds,
    targets,
    max_target_tokens=None,
    save_preds_to=None,
    n_samples=1000,
    good_score=None,
    bad_score=None,
    seed=17,
):

    np.random.seed(seed)
    _preds = preds[:n_samples]
    _targets = targets[:n_samples]

    doc_ids = list(range(len(_preds)))
    results = p_map(
        lambda pred, target, doc_id: eval_job(
            pred,
            target,
            doc_id,
            good_score=good_score,
            bad_score=bad_score,
            max_target_tokens=max_target_tokens,
        ),
        _preds,
        _targets,
        doc_ids,
    )

    results = _aggregate_results(results)
    _print_eval_metrics(results)

    if save_preds_to:
        filepath = pathlib.Path(save_preds_to)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        preds_df = pd.DataFrame({"predictions": _preds, "targets": _targets})
        preds_df.to_csv(save_preds_to)

        results_filename = f"{filepath.stem}_results.txt"
        results_filename = filepath.parent / results_filename
        with open(results_filename, "w") as file:
            file.write(json.dumps(results["scores"], indent=2))

    return results["scores"]


def eval_sampled_job(
    pred_views,
    target,
    text_guidance=None,
    token_budget=None,
    strict_budget=False,
    min_budget=10,
    oracle_budget=False,
    adjust_budget=-30,
    method="factorsum",
    content_weight=1.0,
    budget_weight=1.0,
):

    if oracle_budget:
        target_words = nltk.word_tokenize("\n".join(target))
        _token_budget = len(target_words)
    else:
        _token_budget = token_budget

    _token_budget += adjust_budget
    if _token_budget < min_budget:
        _token_budget = min_budget

    summary, _ = find_best_summary(
        pred_views,
        content_guidance=text_guidance,
        budget_guidance=_token_budget,
        strict_budget=strict_budget,
        content_weight=content_weight,
        budget_weight=budget_weight,
        method=method,
    )

    summary = "\n".join(summary)

    return summary


def evaluate_sampled(
    summary_views,
    targets,
    doc_ids,
    text_guidance=None,
    method="factorsum",
    token_budget=None,
    strict_budget=False,
    min_budget=10,
    max_target_tokens=None,
    good_score=None,
    bad_score=None,
    oracle_budget=False,
    adjust_budget=-30,
    n_samples=1000,
    content_weight=1.0,
    budget_weight=1.0,
    save_preds_to=None,
    seed=17,
):

    unique_doc_ids = list(set(doc_ids))
    _summary_views = []
    _targets = []
    _text_guidance = []
    _token_budget = []

    for doc_id in unique_doc_ids[:n_samples]:
        doc_summary_views = [
            view for d_id, view in zip(doc_ids, summary_views) if d_id == doc_id
        ]
        _summary_views.append(doc_summary_views)
        _targets.append(targets[doc_id])

        if text_guidance is not None:
            _text_guidance.append(text_guidance[doc_id])
        else:
            _text_guidance.append(None)

        if type(token_budget) == list:
            _token_budget.append(token_budget[doc_id])
        else:
            _token_budget.append(token_budget)

    summaries = p_map(
        lambda sample_summary_views, target, text_guidance, token_budget: eval_sampled_job(
            sample_summary_views,
            target,
            text_guidance=text_guidance,
            token_budget=token_budget,
            strict_budget=strict_budget,
            min_budget=min_budget,
            oracle_budget=oracle_budget,
            adjust_budget=adjust_budget,
            content_weight=content_weight,
            budget_weight=budget_weight,
            method=method,
        ),
        _summary_views,
        _targets,
        _text_guidance,
        _token_budget,
    )

    return evaluate(
        summaries,
        targets,
        max_target_tokens=max_target_tokens,
        good_score=good_score,
        bad_score=bad_score,
        n_samples=n_samples,
        seed=seed,
        save_preds_to=save_preds_to,
    )
