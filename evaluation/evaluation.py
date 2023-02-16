import pathlib
import json
import logging

import numpy as np
import pandas as pd
from rouge_score import scoring
from p_tqdm import p_map

from .utils import log_summary, log_scores, word_tokenize, aggregate_scores
from factorsum.utils import apply_word_limit, sent_tokenize
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
    for score_key in scores:
        log_scores(score_key, scores[score_key])


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
    elif "\n" in pred:
        pred_sents = pred.split("\n")
    else:
        pred_sents = sent_tokenize(pred)

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

    pred_words = word_tokenize(pred)
    target_words = word_tokenize(target)

    sents_per_summary_doc = len(pred_sents)
    tokens_per_summary_sent_doc = []
    for sent in pred_sents:
        words = word_tokenize(sent)
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
        target_sents = sent_tokenize(target)
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
    scores=None,
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
    
    scores_df = {}

    if scores:
        for score_key in scores:
            agg_scores = aggregate_scores(scores[score_key])
            results["scores"][score_key] = agg_scores
            
            for sample_scores in scores[score_key]:
                for sub_key, value in sample_scores.items():
                    values = scores_df.get(f'{score_key}_{sub_key}', [])
                    value = sample_scores[sub_key][0]
                    values.append(value)
                    scores_df[f'{score_key}_{sub_key}'] = values

    for scores in results['rouge_scores']:
        for score_key, score in scores.items():
            for sub_key in ['precision', 'recall', 'fmeasure']:
                values = scores_df.get(f'{score_key}_{sub_key}', [])
                value = getattr(score, sub_key)
                values.append(value)
                scores_df[f'{score_key}_{sub_key}'] = values

    _print_eval_metrics(results)        

    if save_preds_to:
        filepath = pathlib.Path(save_preds_to)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        preds_df = pd.DataFrame({"predictions": _preds, "targets": _targets})
        preds_df.to_csv(save_preds_to, index=False)

        scores_df = pd.DataFrame(scores_df)
        scores_filename = f"{filepath.stem}_scores.csv"
        scores_filename = filepath.parent / scores_filename
        scores_df.to_csv(scores_filename, index=False)

        results_filename = f"{filepath.stem}_results.txt"
        results_filename = filepath.parent / results_filename
        with open(results_filename, "w") as file:
            file.write(json.dumps(results["scores"], indent=2))

    return results["scores"]
