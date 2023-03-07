import json
import logging
import os
import pathlib

import datasets
import fire
import numpy as np
import pandas as pd
from p_tqdm import p_map
from rouge_score import scoring

from .inference import predict_summaries
from .utils import (
    aggregate_scores,
    config_logging,
    get_output_path,
    log_summary,
    log_scores,
    sent_tokenize,
    word_tokenize,
)
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


def _get_text_statistics(target, summary):
    sents_per_summary = []
    tokens_per_summary = []
    tokens_per_summary_sent = []
    sents_per_abstract = []
    tokens_per_abstract = []
    tokens_per_abstract_sent = []
    length_diffs = []

    if isinstance(target, list):
        target_sents = target
        target = "\n".join(target)
    else:
        target_sents = sent_tokenize(target)

    if isinstance(summary, list):
        summary_sents = summary
        summary = "\n".join(summary)
    else:
        summary_sents = sent_tokenize(summary)

    pred_words = word_tokenize(summary)
    target_words = word_tokenize(target)

    length_diff = len(pred_words) - len(target_words)
    length_diffs.append(length_diff)
    sents_per_summary.append(len(summary_sents))
    tokens_per_summary_sent.append(np.mean([len(s.split()) for s in summary_sents]))
    tokens_per_summary.append(len(pred_words))
    sents_per_abstract.append(len(target_sents))
    tokens_per_abstract_sent.append(np.mean([len(s.split()) for s in target_sents]))
    tokens_per_abstract.append(len(target_words))

    statistics = dict(
        sents_per_summary=sents_per_summary,
        tokens_per_summary=tokens_per_summary,
        tokens_per_summary_sent=tokens_per_summary_sent,
        sents_per_abstract=sents_per_abstract,
        tokens_per_abstract=tokens_per_abstract,
        tokens_per_abstract_sent=tokens_per_abstract_sent,
        length_diffs=length_diffs,
    )
    return statistics


def eval_job(
    pred,
    target,
    doc_id,
    good_score=None,
    bad_score=None,
):
    rouge_scores = []

    try:
        if target is None or str(target) == "nan" or len(target) == 0:
            return
    except:
        logger.error(f"Invalid target summary: {target}")

    metrics = summarization_metrics(pred, target_summary=target)
    rouge_score = metrics["rouge"]

    if rouge_score:
        rouge_scores.append(rouge_score)

    log_summary(doc_id, pred, target, metrics, bad_score, good_score)

    stats = _get_text_statistics(target, pred)
    results = dict(rouge_scores=rouge_scores, **stats)
    return results


def evaluate(
    preds,
    targets,
    scores=None,
    save_preds_to=None,
    n_samples=None,
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
                    values = scores_df.get(f"{score_key}_{sub_key}", [])
                    if isinstance(sample_scores[sub_key], list):
                        value = sample_scores[sub_key][0]
                    else:
                        value = sample_scores[sub_key]
                    values.append(value)
                    scores_df[f"{score_key}_{sub_key}"] = values

    for scores in results["rouge_scores"]:
        for score_key, score in scores.items():
            for sub_key in ["precision", "recall", "fmeasure"]:
                values = scores_df.get(f"{score_key}_{sub_key}", [])
                value = getattr(score, sub_key)
                values.append(value)
                scores_df[f"{score_key}_{sub_key}"] = values

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


def evaluate_model(
    model_name="google/pegasus-arxiv",
    dataset_path="scientific_papers",
    dataset_name="arxiv",
    split="test",
    source_key="article",
    target_key="abstract",
    max_samples=None,
    output_dir=None,
    cache_start=0,
    cache_end=None,
    cache_dir=None,
    seed=17,
):
    timestr = config_logging(dataset_name, split, output_dir)
    eval_data = datasets.load_dataset(dataset_path, dataset_name, cache_dir=cache_dir)
    eval_data = eval_data[split]
    sources = eval_data[source_key][:max_samples]
    targets = eval_data[target_key][:max_samples]

    logger.info("Reference summary (sanity check)")

    evaluate(
        targets,
        targets,
        n_samples=max_samples,
        seed=seed,
    )

    if isinstance(model_name, (list, tuple)):
        model_names = model_name
    else:
        model_names = [model_name]

    for model_name in model_names:
        logger.info(f"Evaluating {model_name}")
        preds = predict_summaries(
            model_name,
            sources,
            cache_start=cache_start,
            cache_end=cache_end,
        )
        save_to = get_output_path(
            output_dir,
            dataset_name,
            split,
            timestr=timestr,
            custom_suffix=model_name.replace("/", "_"),
        )
        evaluate(
            preds,
            targets,
            save_preds_to=save_to,
            seed=seed,
        )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fire.Fire(evaluate_model)
