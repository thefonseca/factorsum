import pathlib
import json
import logging

import nltk
import numpy as np
import pandas as pd
from rouge_score import scoring
from p_tqdm import p_map
from scipy.stats import bootstrap

from .utils import log_summary, log_rouge_scores
from factorsum.utils import apply_word_limit
from factorsum.metrics import summarization_metrics

logger = logging.getLogger(__name__)


def _print_eval_metrics(results):
    info_str = [
        f"Number of documents: {len(results['sents_per_summary'])}",
        f"Avg sentences per summary: {np.mean(results['sents_per_summary'])}",
        f"Avg tokens per summary: {np.mean(results['tokens_per_summary'])}",
        f"Avg tokens per summary sent: {np.mean(results['tokens_per_summary_sent'])}",
        f"Avg sentences per abstract: {np.mean(results['sents_per_abstract'])}",
        f"Avg tokens per abstract: {np.mean(results['tokens_per_abstract'])}",
        f"Avg tokens per abstract sent: {np.mean(results['tokens_per_abstract_sent'])}",
        f"Avg token difference: {np.mean(results['length_diffs'])}",
    ]

    logger.info("\n".join(info_str))
    scores = results["scores"]
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


def _print_guidance_scores(scores):
    if len(scores) == 0:
        return

    info = ["Guidances scores:"]
    for key in scores.keys():
        if type(scores[key]) == dict:
            _scores = [f"{scores[key][x]:.3f}" for x in ["low", "mean", "high"]]
            _scores = ", ".join(_scores)
        else:
            _scores = f"{scores[key]:.3f}"
        info.append(f"{key}: {_scores}")
    logger.info("\n".join(info))


def _aggregate_guidance_scores(scores):
    if len(scores) == 1:
        return scores[0]
    elif len(scores) == 0:
        return {}

    agg_scores = {}

    for score in scores:
        for key in score.keys():
            key_scores = agg_scores.get(key, [])
            key_scores.append(score[key])
            agg_scores[key] = key_scores

    confidence_intervals = {}
    for key in agg_scores.keys():
        ci = bootstrap(
            (agg_scores[key],),
            np.mean,
            confidence_level=0.95,
            random_state=17,
            method="BCa",
        )
        confidence_intervals[key] = {
            "low": ci.confidence_interval.low,
            "high": ci.confidence_interval.high,
            "mean": np.mean(agg_scores[key]),
        }

    return confidence_intervals


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
    elif "\n" in pred:
        pred_sents = pred.split("\n")
    else:
        pred_sents = nltk.sent_tokenize(pred)

    pred = "\n".join(pred_sents)

    try:
        if target is None or str(target) == "nan" or len(target) == 0:
            return
    except:
        logger.error(f"Invalid target summary: {target}")

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

    metrics = summarization_metrics(pred_sents, target_summary=target)
    rouge_score = metrics["rouge"]

    if rouge_score:
        rouge_scores.append(rouge_score)

    log_summary(doc_id, pred, target, metrics, bad_score, good_score)

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
    guidance_scores=None,
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

    if guidance_scores:
        guidance_scores = _aggregate_guidance_scores(guidance_scores)
        _print_guidance_scores(guidance_scores)
        results["scores"]["guidance_scores"] = guidance_scores

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
